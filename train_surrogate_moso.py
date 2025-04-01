# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
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
import logging
import os
from torch.utils.data import Subset
from image_folder import ImageFolderFilenames
from models import DiT_models
from retrain import random_opt, nopt2, nopt2_keys
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from train_autoencoder import create_vae_encoder
from torchvision.utils import save_image
import datasets
datasets.transforms import build_transforms
datasets import build_dataset
from transport import create_transport
from train import update_ema, requires_grad, get_dataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################






#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = args.global_seed 
    torch.manual_seed(seed)

    # Setup an experiment folder:
    dataset_name = args.dataset_name
    if True:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}_conditional_surrogatemoso_fm_%s" %dataset_name

        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae_pretrained = False
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
    
    fm = False
    if args.transport !='diffusion':
        fm = True
    if not fm:
        diffusion = create_diffusion(timestep_respacing="") 
        learn_sigma = True # default: 1000 steps, linear noise schedule
    else:
        diffusion = create_transport()
        learn_sigma = False
    if vae_pretrained:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        logger.info(f"used pretrained vae")
    else:
        if dataset_name == 'celebhq':
            encoder_path = 'logs_tokenizer/celebhq_taming_kl_f8/models/eval_91000.pt' 
            encoder_type = 'kl_gan_taming'
        else:
            encoder_path = 'logs_tokenizer/llamagen_256/models/eval_100000.pt'
            encoder_type = 'vq_gan_taming'
        vae = create_vae_encoder(encoder_type, encoder_path).to(device) #
        logger.info(f"used retrained vae: {encoder_path}")
    
   
    dataset = get_dataset(name=dataset_name)
    batch_size = int(args.global_batch_size)

    print('finished building dataset')
    
    logger.info(f"Dataset contains {len(dataset):,} images ({args.dataset_name})")

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    number_trials = 12

    support_set_size = len(dataset)//number_trials
    for trial_idx in range(args.trial_start_index, number_trials): #args.trial_start_index+14):
        end_idx = min((trial_idx+1) * support_set_size, len(dataset), (trial_idx+1)*support_set_size)
        subset_support_idx = range(trial_idx *support_set_size, end_idx)
        current_model = DiT_models[args.model](
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=args.num_classes,
            class_dropout_prob = class_dropout_prob, #do not use classifier-free guidance
            learn_sigma = learn_sigma
        ).to(device)
        current_ema = deepcopy(current_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(current_ema, False)
        update_ema(current_ema, current_model, decay=0)  # Ensure EMA is initialized with synced weights
        current_model.train()  # important! This enables embedding dropout for classifier-free guidance
        current_ema.eval() 
        opt = torch.optim.AdamW(current_model.parameters(), lr=1e-4, weight_decay=0)

        subset_support = torch.utils.data.Subset(dataset, subset_support_idx)

        loader = DataLoader(
                    subset_support,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True
                )

        logger.info(f"Training for {args.epochs} epochs...")
    
        for epoch in range(args.epochs):
            logger.info(f"Beginning epoch {epoch}...")
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
                        x = vae.encode(x)['quantized'] 
                        x = x.contiguous() * scale_factor
                if not fm:
                    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                    loss_dict = diffusion.training_losses(current_model, x, t, model_kwargs)
                else:
                    loss_dict = diffusion.training_losses(current_model, x, model_kwargs)
                
                
                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                update_ema(current_ema, current_model)

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
                    avg_loss = avg_loss.item()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
            if True: #train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": current_model.state_dict(),
                    "ema": current_ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/trial{trial_idx}_epoch{epoch}.pt"
                torch.save(checkpoint, checkpoint_path)
                print('saved model at checkpoint %s' %checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                #torch.save(training_noise, os.path.join(*[noise_dir, 'noise_trial%d_epoch%d.pt'%(trial_idx, epoch)]))
                print('done with trial idx %d, epoch %d' % (trial_idx, epoch))


        current_model.eval()  # important! This disables randomized embedding dropout

        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="surrogates")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1_0000)
    parser.add_argument("--class-dropout-prob", type=float, default=0.0)
    parser.add_argument("--transport", type=str, default='diffusion')
    parser.add_argument("--trial_start_index", type=int, default=0)
    args = parser.parse_args()
    main(args)
