"""
Train a diffusion model on images.
"""


import random
import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from collections import defaultdict
from functools import partial


def main(args):
    dataset = 'imagenet'
    if dataset == 'imagenet':
        data_path = 'results/150-DiT-XL-2_fm_conditional_surrogateloss_imagenet/forgetting'
        loss_file = os.path.join(data_path,'loss_0060000.pt')
        cont_str = '0060000'
    elif dataset =='celebhq':
        data_path = 'results/083-DiT-XL-2_surrogateforgettingt120_vaeretrained/forgetting'
        loss_file = os.path.join(data_path,'loss100steps_0060000.pt')  
        loss_file = os.path.join(data_path,'loss100steps_0060000.pt')
        cont_str = '0060000'
    losses_map = torch.load(loss_file)
    need_correction = False
    if need_correction:
        corrected_losses = defaultdict(partial(list))
        for l in losses_map.keys():
            k = int(l)
            corrected_losses[k].append(losses_map[l][0])
        losses_map = corrected_losses
    num_flips = defaultdict(partial(int, 0))
    for k, losses in losses_map.items():
        for l in range(len(losses)-1):
            if losses[l] < losses[l+1]:
                print('loss increased')
                num_flips[k] += 1 
        num_flips[k] = float(num_flips[k])/len(losses)
    print('num_flips', num_flips) #ratio
    torch.save(num_flips, os.path.join(data_path, 'flipratio_%s.pt'%cont_str))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--data-path", type=str, default='pruning/datasets/CelebAMask-HQ')
    parser.add_argument("--experiment_dir", type=str, default="results/005-DiT-XL-2",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)


