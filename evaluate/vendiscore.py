import torch
from torchvision import transforms
from vendi_score import image_utils
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
random.seed(0)
torch.manual_seed(0)
dpaths = ["019-DiT-XL-2_conditional_imagenet_vaeretrained_fm_grad_pruned05"
]

rnd_indices = np.random.choice(12800, 4000, replace=False)
print('starting evaluation...')
for dpath in dpaths1:
    if dpath in dpaths:
        print('already evaluated ', dpath)
        continue
    full_path = "results1/" + dpath +"samples/all_samples_numsamples12800.pth"
    data=torch.load(full_path)
    data = data[rnd_indices]
    data =  [transforms.ToPILImage()(datai) for datai in data]
    inception_vs = image_utils.embedding_vendi_score(data, device="cuda")

    print('vendi score', dpath, inception_vs, "\n\n")

