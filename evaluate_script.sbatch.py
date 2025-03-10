#!/bin/bash
#SBATCH --account= account           
#SBATCH --nodes=1                        # How many compute nodes
#SBATCH --gres=gpu:4 
#SBATCH --job-name=exp_name
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=48              
#SBATCH -o logs/expname%j.log
#SBATCH --time=20:00:00          
#SBATCH --partition=partition

source env_name/activate.sh
model="061-DiT-XL-2_conditional_imagenet_vaeretrained_fm_inversecluster_clip_pruned05"
epoch="0120000"
dataset="celebhq"
if [[ $model == *"imagenet"* ]]; then
  dataset="imagenet"
else
  epoch="0220000"
fi
torchrun --nnodes=1 --nproc_per_node=1 evaluate_fid.py --batch_size 64 --ckpt $model --epoch "$epoch".pt --dataset_name $dataset --num-sampling-steps 250 --num_iters 200



