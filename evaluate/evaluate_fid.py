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
import torch_fidelity

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
    input2 = "pruning/datasets/celebhq/validation"
    
    if 'imagenet' in args.ckpt:
        input2="pruning/datasets/imagenet/validation"
    print('**input2 %s' %input2)
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
    epoch = args.epoch
    results_dir = 'results1' 
    ckpt_path = os.path.join(*[results_dir, args.ckpt, 'checkpoints', epoch])  
    print('**cktp path 1', ckpt_path)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(*['results2', args.ckpt, 'checkpoints', epoch])
        results_dir = 'results2'
    print('**cktp path 2', ckpt_path)
    state_dict = find_model(ckpt_path)
    exp_dir = ckpt_path.split('/')[1]
    epoch = ckpt_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(*[results_dir, exp_dir, 'vis', 'vis4000_%s_seed%d'%(epoch, args.seed)])

    if dataset_name == 'celebhq':
        samples_save_path = os.path.join(save_dir, 'all_samples_numsamples4096.pth')
    
    else:
        samples_save_path = os.path.join(save_dir, 'all_samples_numsamples12800.pth')
    print('samples save path %s' % samples_save_path)
    if True or not os.path.exists(samples_save_path):

        print('***creating samples******')
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

        os.makedirs(save_dir, exist_ok=True)
        print('created dir %s'%save_dir)

        model.load_state_dict(state_dict)
        model.eval()  
        
    
        dataset_name = args.dataset_name
        if vae_pretrained:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        else:
            if dataset_name == 'celebhq':
                encoder_path = 'logs_tokenizer/celebhq_taming_kl_f8/models/eval_91000.pt'
                encoder_type = 'kl_gan_taming'
            else:
                encoder_path = 'logs_tokenizer/llamagen_256/models/eval_100000.pt'
                encoder_type = 'vq_gan_taming'
            print('encoder type %s' %encoder_type)
            vae = create_vae_encoder(encoder_type, encoder_path).to(device) #

        vae.eval()
        batch_size = args.batch_size

        if not is_conditional:
            class_labels = torch.zeros(batch_size).int()
            y = class_labels.to(device)
            n = len(class_labels)
        
        num_ims = 0
        all_samples = []
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

            samples = (((samples*0.5)+0.5)  *255).type(torch.uint8)
            all_samples.append(samples)

        all_samples = torch.cat(all_samples)

        samples_save_path = os.path.join(save_dir, 'all_samples_numsamples%d.pth')%len(all_samples)
        try:
            torch.save(all_samples, samples_save_path)
            print('saved samples')
        except:
            print('exception saving samples')
    else:
        print('***samples already generated, loading...******')
        all_samples = torch.load(samples_save_path).to(device)
    metrics_dict = torch_fidelity.calculate_metrics(
    input1=all_samples, 
    input2=input2, 
    cuda=True, 
    isc=True, 
    fid=True, 
    kid=False, 
    prc=True, 
    verbose=False
    )
    print(metrics_dict)
    print('model used', ckpt_path)
    print('number of samples %d' %len(all_samples))

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
    #parser.add_argument("--ckpt", type=str, default="results/089-DiT-XL-2_vaeretrained_random_pruned05_vae91000/checkpoints/0220000.pt")
    parser.add_argument("--ckpt", type=str, default="089-DiT-XL-2_vaeretrained_random_pruned05_vae91000")
    args = parser.parse_args()
    main(args)


#058-DiT-XL-2_vaeretrained_forgettingpruned025_vae91000
#064-DiT-XL-2_vaeretrained_inverseforgettingpruned025_vae91000
#056-DiT-XL-2_vaeretrained_randompruned025_vae91000
#063-DiT-XL-2_vaeretrained_clusterpruned025
#062-DiT-XL-2_vaeretrained_randompruned06
#59-DiT-XL-2_vaeretrained_vae91000
#067-DiT-XL-2_vaeretrained_l2score_pruned025_vae91000
#065-DiT-XL-2_vaeretrained_gradscore_pruned025_vae91000
#072-DiT-XL-2_vaeretrained_pruned0998temp_vae91000
#070-DiT-XL-2_vaeretrained_moso_pruned025_vae91000
#067-DiT-XL-2_vaeretrained_l2score_pruned025_vae91000
#068-DiT-XL-2_vaeretrained_randompruned099_vae91000
#073-DiT-XL-2_vaeretrained_randompruned_pruned095_vae91000
#075-DiT-XL-2_vaeretrained_inversemoso_pruned095_vae91000
#079-DiT-XL-2_vaeretrained_randompruned09_vae91000
#078-DiT-XL-2_vaeretrained_clusterpruned09_vae91000
#077-DiT-XL-2_vaeretrained_grad_pruned095_vae91000
#076-DiT-XL-2_vaeretrained_moso_pruned095_vae91000
