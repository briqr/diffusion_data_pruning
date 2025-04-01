# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from autoencoder import create_vae_encoder
import os
from train import  get_dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder as ImageFolder

def main(args):
    
    gen_root_path = os.path.join('results', args.feat_path.split('/')[0])

    gt_feat_path = 'features_vaeretrained'
    feat_extractor = 'clip'
    
    if feat_extractor == 'dino':
        train_feat_path = os.path.join(gt_feat_path, 'celebhq_%s_features.pth'%feat_extractor)
        train_feat = torch.load(train_feat_path)
    else:
        train_feat = []
        for s in range(0,1501,500):
            current_f = torch.load(os.path.join(gt_feat_path, 'celebhq_clip_features_%d.pth'%s))
            train_feat += current_f
        current_f = torch.load(os.path.join(gt_feat_path, 'celebhq_clip_features_last.pth'))
        train_feat += current_f
    
    train_feat = torch.from_numpy(np.asarray(train_feat)).float().cuda()
    gen_feat_path = os.path.join(gen_root_path, 'features')
    gen_file = os.path.join(gen_feat_path, 'celebhq_%s_features.pth'%feat_extractor)

    gen_feat = torch.load(gen_file)

    dists_save_path = os.path.join(gen_feat_path, 'distance_gen_train_feat_%s.pth'%feat_extractor)
    dists = torch.cdist(gen_feat.flatten(1), train_feat.flatten(1))
    torch.save(dists,dists_save_path)
    min_i = np.unravel_index(torch.argmin(dists).cpu(), dists.shape)

    args_min = dists.argmin(dim=-1)
    
    print('****average of nearest sample distance***', dists.min(dim=-1)[0].mean())
    print('****median of nearest sample distance***', dists.min(dim=-1)[0].median())
    print('generated features path %s using %s'%(gen_feat_path, feat_extractor))
    print('****************************************')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feat_path", type=str, default="089-DiT-XL-2_vaeretrained_random_pruned05_vae91000")
    args = parser.parse_args()
    main(args)


