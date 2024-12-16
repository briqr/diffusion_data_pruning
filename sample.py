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
import json
from transport import create_transport
from transport import Sampler as diffusion_sampler
from tqdm import tqdm
from torchvision.utils import save_image

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_idx = json.load(open("data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    print('model is ', args.ckpt)
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    dataset_name = args.dataset_name
    if 'vaeretraine' in args.ckpt:
        vae_pretrained = False
    else:
        vae_pretrained = True
    is_conditional = False
    if 'conditional' in args.ckpt and not 'unconditional' in args.ckpt:
        is_conditional = True
    
    if vae_pretrained or dataset_name == 'celebhq':
        in_channels = 4
        scale_factor = 0.18215
    else:
        in_channels = 8
        scale_factor = 0.82608986375
    
    is_conditional = False
    class_dropout_prob = 0
    num_classes = args.num_classes
    if dataset_name =='imagenet':
        is_conditional = True
        class_dropout_prob = 0.1
        num_classes = 1000
    fm = False
    learn_sigma = True
    if '_fm_' in args.ckpt or 'flowmatch' in args.ckpt:
        fm = True
        learn_sigma = False
        transport = create_transport()
        transport_sampler = diffusion_sampler(transport).sample_ode(
        sampling_method="euler",
        num_steps=60,
        )
    else:
        diffusion = create_diffusion(str(args.num_sampling_steps))

    model_type = args.model
    if 'DiT-B-2' in args.ckpt:
        model_type = 'DiT-B/2'
    print('***model type %s' % model_type)
    model = DiT_models[model_type](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob = class_dropout_prob,
        learn_sigma = learn_sigma
    ).to(device)

    epoch = args.epoch
    results_dir = 'results'
    ckpt_path = os.path.join(*[results_dir, args.ckpt, 'checkpoints', epoch])
   
    state_dict = find_model(ckpt_path)
    exp_dir = ckpt_path.split('/')[1]
    epoch = ckpt_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(*[results_dir, exp_dir, 'vis_single', 'vis512_%s_seed%d'%(epoch, args.seed)])
    os.makedirs(save_dir, exist_ok=True)
    print('created dir %s'%save_dir)

    model.load_state_dict(state_dict)
    model.eval()  # important!
    
 
    dataset_name = args.dataset_name
    if vae_pretrained:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    else:
        if dataset_name == 'celebhq':
            encoder_path = 'models/eval_91000.pt' 
            encoder_type = 'kl_gan_taming'
        else:
            encoder_path = 'models/eval_100000.pt'
            encoder_type = 'vq_gan_taming'
        vae = create_vae_encoder(encoder_type, encoder_path).to(device) #

    vae.eval()
    batch_size = args.batch_size

    if not is_conditional:
        class_labels = torch.zeros(batch_size).int()
        y = class_labels.to(device)
        n = len(class_labels)
    
    num_ims = 0
    for s in tqdm(range(args.num_iters)):
        if is_conditional:
            class_labels = torch.randint(0, num_classes, [batch_size])
            n = len(class_labels)
            z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
            y = class_labels.to(device)
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            forward_fn = model.forward_with_cfg
            
        else:
            z = torch.randn(n, in_channels, latent_size, latent_size, device=device)
            model_kwargs = dict(y=y)
            forward_fn = model.forward
        
        if not fm:
            samples, _ = diffusion.p_sample_loop(
                forward_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )

        else:
            samples = transport_sampler(
                z, forward_fn, **model_kwargs
            )[-1]

        if is_conditional:
            samples, _ = samples.chunk(2, dim=0)
        if not vae_pretrained:
            samples = vae.decode(samples / scale_factor) 
        else:
            samples = vae.decode(samples / scale_factor).sample

        im_infile = min(12, len(samples))
        im_infile = 1
        for idx, sample in enumerate(samples):
            save_path = os.path.join(save_dir, format(num_ims, '05d')+'_%s.png'%exp_dir)
            if num_ims % batch_size==0:
                print('saved samples %s'%save_path)
            l_str = idx2label(class_labels[idx].numpy())
            save_image(sample, save_path, normalize=True, value_range=(-1, 1))
            num_ims += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--dataset_name", type=str, default='celebhq')
    parser.add_argument("--seed", type=int, default=0) 
    parser.add_argument("--epoch", type=str, default='0220000.pt') 
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default="089-DiT-XL-2_vaeretrained_random_pruned05_vae91000")
    args = parser.parse_args()
    main(args)
