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
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from lpips import LPIPS
from discriminator import Discriminator
from torchvision.utils import make_grid
import torchvision
from any_diffusion.autoencoders.taming.kl_gan import KLAutoEncoder
import any_diffusion.autoencoders
from timm import create_model
#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_vae_encoder(model_name, path):
    vae = create_model(
    model_name=model_name,  path=path, pretrained=True)
    return vae