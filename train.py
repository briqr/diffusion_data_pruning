# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
possible to train with or without pruning
Library built on top of 
transormer diffusion model: https://github.com/facebookresearch/DiT
ode: https://github.com/willisma/SiT
autoencoders: https://github.com/CompVis/taming-transformers
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
from torch.utils.data import Subset
from image_folder import ImageFolderFilenames
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from autoencoder import create_vae_encoder
from torchvision.utils import save_image
import datasets
from datasets.transforms import build_transforms
from datasets import build_dataset
from pruning_clustering import furthestnearest_cluster, balanced_cluster, mid_cluster
from transport import create_transport
import logging


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def nopt2(scores, pr, largest=True, is_mid=False):
    # S: dict
    # size: the subset size
    num_samples = int(len(scores) * pr )
    print('nopt2 number of samples', num_samples)
    pool = torch.tensor(scores)
    pool = pool.squeeze()
    if is_mid:
        pruning_ratio = (1 - pr)/2
        num_pruning_largest = int(len(scores) * pruning_ratio)
    else: 
        num_pruning_largest = 0
    index_all = pool.topk(num_samples+num_pruning_largest, largest=largest)[1]
    index_all = index_all[num_pruning_largest:]
    return index_all


#the scores order does not correspond to indices

def nopt2_keys(scores_map, pr, largest=True, is_mid=False):
    num_samples = int(len(scores_map) * pr )
    sorted_ratios = sorted(scores_map.items(), key = lambda x:x[1], reverse=largest)
    top_keys = []
    if is_mid:
        start_index = int(len(scores_map) * (1 - pr)/2)
    else:
        start_index = 0
    for key, val in sorted(sorted_ratios)[start_index:num_samples]:
        top_keys.append(key)
    return top_keys




def random_opt(len_d, pr):
    # S: dict
    # size: the subset size
    print('random selecting....')
    subset_size = int(len_d * pr )
    print('**************subset size', subset_size)
    #size = 32
    pool = torch.rand(len_d)
    print('pool shape, size', pool.shape, subset_size)
    
    index = pool.topk(subset_size)
    return index


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

#splits: "train", "validation", "test"
def get_dataset(split='train', is_training = True, name ='celebhq'):
    if name == 'celebhq':
        cache_dir = 'DATASETS/'
        dataset_name = 'jxie/celeba-hq'
    elif name == 'imagenet':
        cache_dir = 'DATASETS/imagenet-1k'
        dataset_name = 'imagenet-1k'

    train_data_config = {'path': dataset_name,
                    'cache_dir': cache_dir, 
                    'split': split,
                    'trust_remote_code': True                    }

    transforms_config = {'no_aug': True,
                            'is_training': is_training,
                            'input_size': 256,
                            'mean': [0.5, 0.5, 0.5],
                            'std': [0.5, 0.5, 0.5]
                            }

    data_transform = build_transforms(transforms_config, dataset_name=dataset_name)
    print('building dataset')
    dataset = build_dataset(
        train_data_config,
        transforms=data_transform,
    )
    return dataset


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    dataset_name = args.dataset_name
    pr = args.pruning_ratio
    is_pruned = pr > 0.00001
    method = args.pruning_method
    inverse = args.inverse > 0.00001
    is_mid = args.mid > 0.00001
    fm = False
    if args.transport !='diffusion':
        fm = True
    if not fm:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        learn_sigma = True
    else:
        diffusion = create_transport()
        learn_sigma = False
    vae_pretrained = args.vae_pretrained > 0.00001
    
    is_conditional = False
    class_dropout_prob = 0
    if dataset_name =='imagenet':
        is_conditional = True
        class_dropout_prob = 0.1
    if vae_pretrained or dataset_name == 'celebhq':
        in_channels = 4
        scale_factor = 0.18215
    else:
        in_channels = 8
        scale_factor = 0.82608986375

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        pr_str = str(pr).replace('.', '')
        method_str = method
        if inverse > 0:
            method_str =  'inverse' + method
        if is_mid:
            method_str += '_mid'
        if fm:
            method_str = 'fm_' + method_str
        if vae_pretrained:
            method_str = 'vaePretrained_' + method_str
        else:
            method_str = 'vaeretrained_' + method_str
        if not is_conditional:
            method_str = 'unconditional' + method_str
        else:
            method_str = 'conditional' + method_str
        if args.resume:
            exp = args.resume.split('/')[1]
            experiment_dir = f"{args.results_dir}/{exp}"
        else:
            #experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}_conditional_%s_%s_pruned%s"%(dataset_name, method_str, pr_str)  # Create an experiment folder
            experiment_dir = f"{args.results_dir}/%s_%s_pruned%s"%(dataset_name, method_str, pr_str)  # Create an experiment folder
        selected_index_path = os.path.join(experiment_dir, 'selected_index.pth')
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        save_dir = os.path.join(experiment_dir, 'train_images')
        #os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        print(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8




    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob = class_dropout_prob, #do not use classifier-free guidance
        learn_sigma = learn_sigma
    )

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        logger.info(f"Using checkpoint: {args.resume}")
        print(f"Using checkpoint: {args.resume}")

    
    model = DDP(model.to(device), device_ids=[rank])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.resume:
        opt.load_state_dict(checkpoint['opt'])
        del checkpoint
    if vae_pretrained:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        logger.info(f"used pretrained vae")
        print(f"used pretrained vae")
    else:
        if dataset_name == 'celebhq':
            encoder_path = 'models/eval_91000.pt' 
            encoder_type = 'kl_gan_taming'
        else:
            encoder_path = 'models/eval_100000.pt'
            encoder_type = 'vq_gan_taming'
        print('encoder type %s' %encoder_type)
        vae = create_vae_encoder(encoder_type, encoder_path).to(device) #
        logger.info(f"used retrained vae: {encoder_path}")
        print(f"used retrained vae: {encoder_path}")


    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print('args', args)
    logger.info('args', args)
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):


    dataset = get_dataset(name=dataset_name)
    if is_pruned:
        if method == 'forgetting':  #pruning based on forgetting
            score_dir =  f"{args.score_dir}/forgetting"
            #
            if dataset_name == 'celebhq':
                score_dir = "results/083-DiT-XL-2_surrogateforgettingt120_vaeretrained/forgetting"
                score_file_path = os.path.join(score_dir, 'flip_0060000.pt') #celebhq
            else:
                score_file_path = os.path.join(score_dir, 'flipratio_0080000.pt')
            flip_ratio = torch.load(score_file_path, map_location='cpu')
            selected_index = nopt2_keys(flip_ratio, 1-pr, largest=not inverse, is_mid=is_mid) #largest=Tre, return those with maximum number of flips
        elif method=='random': # pruning based on random
            if not args.resume:                      
                selected_index = random_opt(len(dataset), 1-pr)[1]
                if rank == 0:
                    torch.save(selected_index, selected_index_path)
            else:
                exp = args.resume.split('/')[1]
                experiment_dir = f"{args.results_dir}/{exp}"
                selected_index_path = os.path.join(experiment_dir, 'selected_index.pth')
                selected_index = torch.load(selected_index_path, map_location='cpu')
            logger.info("selected indices for training", selected_index)
        elif 'cluster_dino' in method': # pruning based on cluster
            if dataset_name == 'celebhq':
                cluster_path = 'featurescluster_vaeretrained_dino_24.pth'
            else:
                cluster_path = 'pruning/datasets/imagenet/dino_features/train/cluster_dino_1000.pth'
            if method =='cluster_dino':
                if is_mid:
                    selected_index = mid_cluster(1-pr, cluster_path)
                else:
                    selected_index = furthestnearest_cluster(1-pr, cluster_path, largest=not inverse)
            elif method =='cluster_dino_balanced':
                selected_index = balanced_cluster(cluster_path, largest=not inverse)
            
        elif 'cluster_clip' in method: # pruning based on cluster
            if dataset_name == 'celebhq':
                cluster_path = 'features_vaeretrained/cluster_vaeretrained_clip_24.pth'
            else: #imagenet
                cluster_path = 'pruning/datasets/imagenet/clip_features/train/cluster_clip_1000.pth' 
            if cluster_method == 'cluster_clip'
                if is_mid:
                    selected_index = mid_cluster(1-pr, cluster_path)
                else:
                    selected_index = furthestnearest_cluster(1-pr, cluster_path, largest=not inverse)
            elif method =='cluster_clip_balanced': 
                selected_index = balanced_cluster(cluster_path, largest=not inverse)
        
        elif method=='moso': 
            score_dir = 'surrogates/061-DiT-XL-2_surrogatemoso_vaeretrained_vae91000/scores'
            score_file_path = os.path.join(score_dir, 'moso_score.pth') 
            scores = torch.load(score_file_path, map_location='cpu')
            selected_index = nopt2(scores, 1-pr, largest=not inverse, is_mid=is_mid)
        elif method =='grad': 
            if dataset_name =='celebhq':
                score_dir = 'results/083-DiT-XL-2_surrogateforgettingt120_vaeretrained/scores' 
                score_file_path = os.path.join(score_dir, 'score_grad_epoch60.pth')
            else:
                score_dir = 'results/150-DiT-XL-2_fm_conditional_surrogateloss_imagenet/scores' 
                score_file_path = os.path.join(score_dir, 'score_%s_iter60k.pth'%method)
            scores = torch.load(score_file_path, map_location='cpu')
            print('** len of scores, ', len(scores) )
            selected_index = nopt2(scores, 1-pr, largest=not inverse, is_mid=is_mid)
            print('len of selected index')
        elif method=='l2loss': 
            if dataset_name == 'celebhq':
                score_dir = 'results/083-DiT-XL-2_surrogateforgettingt120_vaeretrained/scores' 
                score_file_path = os.path.join(score_dir, 'score_l2_fixed120timestep_epoch60.pth') 
            else:
                score_dir = 'results/150-DiT-XL-2_fm_conditional_surrogateloss_imagenet/scores' 
                score_file_path = os.path.join(score_dir, 'score_l2_iter80k.pth')
                
            scores = torch.load(score_file_path, map_location='cpu')
            selected_index = nopt2(scores, 1-pr, largest=not inverse, is_mid=is_mid)
        #prune the dataset
        dataset = torch.utils.data.Subset(dataset, selected_index) 

        print('finished building dataset')
    
    print('creating dataloader')
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")
    print(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")

    train_steps = 0
    log_steps = 0
    running_loss = 0

    # Prepare models for training:
    if args.resume:
        train_steps = int(args.resume.split('/')[-1].split('.')[0])
        init_start_steps = train_steps
        start_epoch = int(train_steps / (len(dataset) / args.global_batch_size))
        logger.info(f"Initial state: step={train_steps}, epoch={start_epoch}")
        print(f"Initial state: step={train_steps}, epoch={start_epoch}")
    else:
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights 
        start_epoch = 0
        init_start_steps = 0
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode


    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        num_ims = 0
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        print(f"Beginning epoch {epoch}...")
        for batch in loader:

            x = batch['image']
            y = batch['label']
            if not is_conditional:
                y[...] = 0
            x = x.to(device)
            y = y.to(device)
            model_kwargs = dict(y=y)
            if vae_pretrained:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(scale_factor)
            else:
                with torch.no_grad():
                    if dataset_name == 'imagenet':                    
                        x = vae.encode(x)['quantized'] # vq gan quantized
                    else:
                        x = vae.encode(x)['quantized'].sample()
                    x = x.contiguous() * scale_factor
            if not fm:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            else:
                loss_dict = diffusion.training_losses(model, x, model_kwargs)
            
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    print(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            if train_steps % 30000 ==0 and init_start_steps %30000 ==0:
                print('1 number of steps divides by 300000,  breaking...')
                break
        if train_steps % 30000 ==0 and init_start_steps %30000 ==0:
            print('2 number of steps divides by 300000,  breaking...')
            break
    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=str)
    parser.add_argument("--results-dir", type=str, default="results2")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1_0000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--pruning_method", type=str, default='random')
    parser.add_argument("--pruning_ratio", type=float, default=0)
    parser.add_argument("--inverse", type=int, default=0)
    parser.add_argument("--transport", type=str, default='fm')
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--vae_pretrained", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
