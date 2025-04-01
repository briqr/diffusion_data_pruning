from PIL import Image
import requests

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from torchvision import transforms
from train import center_crop_arr
import torch
from train import get_dataset
import os
import numpy as np 
from torchvision.datasets import ImageFolder as ImageFolder
from tqdm import tqdm 
import argparse

data_type = 'gen' #real

device = "cuda" if torch.cuda.is_available() else "cpu"
print('using device ', device)
dataset_name = 'celebhq' #celebhq or imagenet
def main(args):
    if data_type == 'real':
        dataset = get_dataset(name=dataset_name, split='validation', is_training=False)
        save_path = 'pruning/datasets/%s/clip_features/validation'%dataset_name
        os.makedirs(save_path, exist_ok=True)
    else:
        data_path = "results/" + args.data_path
        data=torch.load(data_path)
        data =  [transforms.ToPILImage()(datai) for datai in data]
        root_path = os.path.join('results', args.data_path.split('/')[0])
        save_path = os.path.join(root_path, 'features')
        os.makedirs(save_path, exist_ok=True)

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


   
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    print('loaded the clip model, starting to extract the features...')
    
    inputs = processor(images=data, return_tensors="pt", do_rescale=True).to(device)
    with torch.no_grad():
        final_img_features = model.get_image_features(**inputs)
    torch.save(final_img_features, os.path.join(save_path, '%s_clip_features.pth')%dataset_name)
    print('***len of saved image feat shape', len(final_img_features) )
    print('*************************')              

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="089-DiT-XL-2_vaeretrained_random_pruned05_vae91000")
    args = parser.parse_args()
    main(args)
