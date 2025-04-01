"""
Train a diffusion model on images.
"""


import random
import numpy as np
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torch.utils.data import Subset
from image_folder import ImageFolderFilenames
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import copy
from train import center_crop_arr
from autoencoder import create_vae_encoder
from tqdm import tqdm
from torch.autograd import grad
from download import find_model
import glob
import collections
from collections import defaultdict
from diffusion.gaussian_diffusion import mean_flat
from functools import partial
datasets.transforms import build_transforms
datasets import build_dataset
from train import get_dataset
from transport import create_transport


def MoSo_scoring_exact(model, vae, diffusion, dataloader, device, trial_idx, epoch, score_dir, is_conditional, scale_factor, vae_pretrained, fm):
    overall_grad = 0
    model_params = [ p for p in model.parameters() if p.requires_grad ]
    i = 0
    noises = []
    ts = []
    x0=[]
    xt=[]
    ut = []
    for l, batch in enumerate(tqdm(dataloader)):
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
                x = vae.encode(x)['quantized'] # vq gan quantized
                x = x.contiguous() * scale_factor
        
        if not fm:
            noise = torch.randn_like(x)
            noises.append(noise)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            ts.append(t)
            loss_dict = diffusion.training_losses(model, x, t, noise=noise, model_kwargs=model_kwargs)
        else:
            loss_dict = diffusion.training_losses(model, x, model_kwargs)
            ts.append(loss_dict['t'])
            x0.append(loss_dict['x0'])
            xt.append(loss_dict['xt'])
            ut.append(loss_dict['ut'])
        loss = loss_dict["loss"].mean()

        g = list(grad(loss, model_params, create_graph=False, allow_unused=True))
        g = [ p for p in g if p is not None ]
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.to(torch.float32)
        g = g.detach()
        overall_grad = overall_grad * i/(i+1) + g / (i+1)

        N = i+1
        i += 1


    overall_grad = overall_grad
    
    score_list = []
    print('***********finished part 1, starting part 2, trial %d, epoch %d' %(trial_idx, epoch))
    i = -1

    for l, batch in enumerate(dataloader):
        x = batch['image']
        y = batch['label']
        y[...] = 0
        x = x.to(device)
        y = y.to(device)
        if vae_pretrained:
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(scale_factor)
        else:
            with torch.no_grad():                    
                x = vae.encode(x)['quantized'] # vq gan quantized
                x = x.contiguous() * scale_factor
        
        if not fm:
            loss_dict = diffusion.training_losses(model, x, t=ts[l], noise = noises[l], model_kwargs=model_kwargs)
        else:
            loss_dict = diffusion.training_losses(model, x, model_kwargs, t=ts[l], x0=x0[l], xt=xt[l], ut=ut[l])

        loss = loss_dict["loss"].mean()


        g = list(grad(loss, model_params, create_graph=False, allow_unused=True))
        g = [ p for p in g if p is not None ]
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()

        score = (2*N-3)/(N-1)**2 * (overall_grad * overall_grad).sum() - 1/(N-1)**2 * (g * g).sum() + (2*N - 4)/(N-1)**2 * ((overall_grad - 1/N * g) * g).sum()
        score = score.detach().cpu()#.numpy()
        score_list.append(score)

    score_list = torch.tensor(score_list).detach()
    score_path = os.path.join(score_dir, f"scores_trial{trial_idx}_epoch{epoch}.pth")
    torch.save(score_list, score_path)
    return score_list


def main(args):
    
    torch.manual_seed(args.seed)
    #torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_path = args.experiment_dir 
    checkpoint_dir =  f"{exp_path}/checkpoints" 
    score_dir = f"{exp_path}/scores"  # Stores saved model checkpoints
    os.makedirs(score_dir, exist_ok=True)



    dataset_name = args.dataset_name
    print('1 dataset name is %s' %dataset_name)
    fm = False
    if '_fm_' in checkpoint_dir:
        fm = True
    if not fm:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        learn_sigma = True
    else:
        diffusion = create_transport()
        learn_sigma = False

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae_pretrained = True
    is_conditional = False
    class_dropout_prob = 0
    num_classes = args.num_classes
    if dataset_name =='imagenet' or 'imagenet' in exp_path:
        is_conditional = True
        class_dropout_prob = 0.1
        num_classes = 1000
        dataset_name = 'imagenet'
    if vae_pretrained or dataset_name == 'celebhq':
        in_channels = 4
        scale_factor = 0.18215
    else:
        in_channels = 8
        scale_factor = 0.82608986375
    print('dataset is %s' %dataset_name)


    
    if vae_pretrained:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        if dataset_name == 'celebhq':
            print('********** 192 dataset is celenhq')
            encoder_path = 'ogs_tokenizer/celebhq_taming_kl_f8/models/eval_91000.pt' #eval_55000.pt,  eval_108000.pt
            encoder_type = 'kl_gan_taming'
        else:
            print('********** 196 dataset is imagenet')
            encoder_path = 'logs_tokenizer/llamagen_256/models/eval_100000.pt'
            encoder_type = 'vq_gan_taming'
        vae = create_vae_encoder(encoder_type, encoder_path).to(device) #





    
    dataset = get_dataset(name=dataset_name)


    
    number_trials = 120


    ckpt_file_names = os.path.join(checkpoint_dir, '*.pt')
    # get all saved ckpt file names
    ckpt_file_list = glob.glob(ckpt_file_names)

    latent_size = args.image_size // 8
    #scores_all_trials = defaultdict(partial(int, 0))
    scores_all_trials = 0
    support_set_size = len(dataset)//number_trials
    for trial_idx in range(0, number_trials):
        start_idx = trial_idx *support_set_size
        end_idx = min((trial_idx+1) * support_set_size, len(dataset), (trial_idx+1)*support_set_size)
        subset_support_idx = range(start_idx, end_idx)

        subset_support = torch.utils.data.Subset(dataset, subset_support_idx)

       
        loader = DataLoader(
            subset_support,
            batch_size=1,
            shuffle=False,
            num_workers=24,
            pin_memory=True,
            drop_last=True
        )

        trial_name = 'trial' + str(trial_idx)+'_'
        trial_file_list = []
        support_mask = np.zeros(len(dataset), dtype=int)
        support_mask[start_idx:end_idx] = 1
        query_mask = support_mask
        for file_name in ckpt_file_list:
            if file_name.find(trial_name)!=-1:
                trial_file_list.append(file_name)
        num_sampels = 10
        trial_file_list = random.sample(trial_file_list, num_sampels)
        print('list of random epochs of trial %d' %trial_idx, trial_file_list)
        epoch = -1
        trial_scores = 0
        for model_path in trial_file_list:
            print('****scoring ', model_path, trial_idx)
            epoch = int(model_path.split('epoch')[1].split('.')[0])
            epoch_scores_path = os.path.join(score_dir, f"score{trial_idx}_epoch{epoch}.pth")
            if False and os.path.exists(epoch_scores_path):
                print('epoch scores path already exists %s', epoch_scores_path)
                scores = torch.load(epoch_scores_path)
                continue

            else:

                model = DiT_models[args.model](
                input_size=latent_size,
                in_channels=in_channels,
                num_classes=num_classes,
                class_dropout_prob = class_dropout_prob, #do not use classifier-free guidance
                learn_sigma = learn_sigma
                ).to(device)
                
                state_dict = find_model(model_path)
                model.load_state_dict(state_dict)
                model.eval() 
                                            
                scores = MoSo_scoring_exact(model, vae, diffusion, loader, device, trial_idx, epoch, score_dir, is_conditional, scale_factor, vae_pretrained, fm)

            print('trial idx %d epoch %d trial scores before '%(trial_idx, epoch), scores, len(scores) )

            trial_scores = trial_scores + scores
            demasked_scores = []
            flag = -1
            for query_index in range(len(query_mask)):
                if query_mask[query_index]==1:
                    flag = flag + 1
                    demasked_scores.append(trial_scores[flag])
                else:
                    demasked_scores.append(0)
            demasked_scores = torch.tensor(demasked_scores).cpu().numpy()
            scores_all_trials = scores_all_trials + demasked_scores            

    score_save_path = os.path.join(score_dir, 'moso_score.pth')
    torch.save(scores_all_trials, score_save_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--experiment_dir", type=str, default="surrogates/conditional_surrogatemoso_fm_imagenet") 
    args = parser.parse_args()
    main(args)


