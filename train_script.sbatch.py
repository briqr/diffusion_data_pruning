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
dataset="imagenet"
pruning_method="cluster_clip"
pruning_ratio="0.5"
num_classes="1000"
batch_size="256"
inverse="1" #inverse indicates nearest to to the center
mid="0"
vae_pretrained="0"
srun torchrun --nnodes=1 --nproc_per_node=4 train.py --dataset $dataset --pruning_method $pruning_method --pruning_ratio $pruning_ratio --inverse $inverse --mid $mid --global-batch-size $batch_size --num-classes $num_classes --vae_pretrained $vae_pretrained