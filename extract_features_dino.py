"""
Author:Purnasai
Description:This file generates image features from
        Database of images & stores them h5py file.
"""
import os
import random

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder as ImageFolder
from retraindit_vaeretrained import get_dataset


dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
model_path = 'libs/dinov2/pretrained/dinov2_vits14.pth'

import ssl
datasets.transforms import build_transforms
datasets import build_dataset

@torch.no_grad()



data_name = 'train'
image_size = 266
dataset_name = 'imagenet'

root_path = '089-DiT-XL-2_vaeretrained_random_pruned05_vae91000'

if data_name == 'train':
    dataset = get_dataset(name=dataset_name)
    save_path = "datasets/imagenet/dino_features/train"
elif data_name == 'gen':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    data_path = os.path.join(root_path, 'vis')
    dataset = ImageFolder(data_path, transform=transform)
    save_path = os.path.join(root_path, 'features')
    os.makedirs(save_path, exist_ok=True)

train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
final_img_features = []
final_img_filepaths = []
idx = 0
chunk = 0
for idx, batch in enumerate(tqdm(train_dataloader)):
    print('idx now %d' %idx)
    if True:
        if data_name == 'train':
            image_tensors = batch['image'].cuda()
        else:
            image_tensors = batch[0].cuda()
        with torch.no_grad():
            image_features = dinov2_vitl14(image_tensors) #384 small, #768 base, #1024 large
        image_features /= image_features.norm(dim=-1, keepdim=True)

        final_img_features += image_features.cpu()


torch.save(final_img_features, os.path.join(save_path, '%s_dino_features.pth')%(dataset_name))              
