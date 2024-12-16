# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
#evaluate samples score for GraNd and EL2N pruning methods
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
from torch.autograd import grad
from download import find_model
from retraindit_vaeretrained import get_dataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


#################################################################################
#                                  Training Loop                                #
#################################################################################


def l2_grad_scores(model, vae, diffusion, dataloader, scale_factor, is_conditional, vae_pretrained, fm, device):
losses = []
gradients = []
model_params = [ p for p in model.parameters() if p.requires_grad ]
for l, batch in enumerate(dataloader):
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
        t[...] = 15
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
    else:
        loss_dict = diffusion.training_losses(model, x, model_kwargs)

    loss = loss_dict["loss"].mean()
    losses.append(loss.item())

    g = list(grad(loss, model_params, create_graph=False, allow_unused=True))
    g = [ p for p in g if p is not None ]
    g = torch.nn.utils.parameters_to_vector(g)
    g = g.to(torch.float32)
    g = g.detach()
    grad_norm = torch.norm(g).item()
    gradients.append(grad_norm)
gradients = torch.tensor(gradients).detach()
score_list = torch.tensor(losses).detach()
return score_list, gradients



def main(args):
"""
Trains a new DiT model.
"""

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
exp_path = args.experiment_dir 
checkpoint_dir =  f"{exp_path}/checkpoints" 
score_dir = f"{exp_path}/scores"  # Stores saved model checkpoints
print('score dir is %s' %score_dir)
os.makedirs(score_dir, exist_ok=True)
# Setup an experiment folder:
dataset_name = args.dataset_name
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
model = DiT_models[args.model](
    input_size=latent_size,
    in_channels=in_channels,
    num_classes=num_classes,
    class_dropout_prob = class_dropout_prob, #do not use classifier-free guidance
    learn_sigma = learn_sigma
).to(device)


if vae_pretrained:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
else:
    if dataset_name == 'celebhq':
        encoder_path = 'logs_tokenizer/celebhq_taming_kl_f8/models/eval_91000.pt' #eval_55000.pt,  eval_108000.pt
        encoder_type = 'kl_gan_taming'
    else:
        encoder_path = 'logs_tokenizer/llamagen_256/models/eval_100000.pt'
        encoder_type = 'vq_gan_taming'
    vae = create_vae_encoder(encoder_type, encoder_path).to(device) #






dataset = get_dataset(name=dataset_name)
print('finished building dataset')
end_index = min(args.start_index+100000, len(dataset))
selected_index = np.arange(args.start_index, end_index)
dataset = torch.utils.data.Subset(dataset, selected_index) 
print('dataset length is %d' %len(dataset), 'start index is %d' %args.start_index)
print('creating dataloader')
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

model_path = os.path.join(checkpoint_dir, '0060000.pt') 


state_dict = find_model(model_path)
model.load_state_dict(state_dict)
model.eval() 

if True:
    print('aggregating results')
    l2scores, grad_scores = l2_grad_scores(model, vae, diffusion, dataloader, scale_factor, is_conditional, vae_pretrained, fm, device)
    score_path = os.path.join(score_dir, 'score_l2_iter60k_%d.pth'%args.start_index)
    torch.save(l2scores, score_path)
    score_path = os.path.join(score_dir, 'score_grad_iter60k_%d.pth'%args.start_index)
    torch.save(grad_scores, score_path)

scores1 = []
scores2 = []
for s in range(0, 1300000, 100000):
    path1 = os.path.join(score_dir, 'score_l2_iter60k_%d.pth'%s)
    current_score1 = torch.load(path1)
    scores1.append(current_score1)
    path2 = os.path.join(score_dir, 'score_grad_iter60k_%d.pth'%s)
    current_score2 = torch.load(path2)
    scores2.append(current_score2)
scores1 = torch.cat(scores1)
scores2 = torch.cat(scores2)

torch.save(scores1, os.path.join(score_dir, 'score_l2_iter60k.pth'))
torch.save(scores2, os.path.join(score_dir, 'score_grad_iter60k.pth'))

if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_dir", type=str, default="results/150-DiT-XL-2_fm_conditional_surrogateloss_imagenet")
parser.add_argument("--results-dir", type=str, default="results")
parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
parser.add_argument("--dataset_name", type=str, default='celebhq')
parser.add_argument("--num-classes", type=int, default=1)
parser.add_argument("--epochs", type=int, default=500000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--ckpt-every", type=int, default=1_0000)
parser.add_argument("--class-dropout-prob", type=float, default=0.0)
parser.add_argument("--start_index", type=int, default=0)
args = parser.parse_args()
main(args)
