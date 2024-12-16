
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
from collections import OrderedDict



#################################################################################
#                             clustering-based pruning Functions                         #
#################################################################################





#furthest or nearest samples with respect to distance from the cluster center
def furthestnearest_cluster(pr, cluster_path, largest=True): # return pr of the cluster samples randomly

    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    print('**feat shape', feat.shape)
    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * (pr) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=largest)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx


# equal number of samples from each cluster, where this number is determined by the number of samples in the smallest cluster
def balanced_cluster(cluster_path, largest=True): # return pr of the cluster samples randomly
   
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]

    all_samples_idx = []
    min_size = 100000
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        if len(cluster_sample_idx) < min_size:
            min_size = len(cluster_sample_idx)
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = min_size
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=largest)[1]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx

#samples from the middle region of the cluster
def mid_cluster(pr, cluster_path): 
   
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]
    all_samples_idx = []
    pruning_pr = (1-pr)/2
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * (pr+pruning_pr) ) 
        dist = torch.norm(feat[cluster_sample_idx] - centers[l], dim=1)
        index = dist.topk(num_samples, largest=True)[1]
        number_furthest =  int(len(cluster_sample_idx) * (pruning_pr))
        index = index[number_furthest:]
        all_samples_idx.extend(cluster_sample_idx[index])

    return all_samples_idx


# return samples in random proximity in the cluster, proportional to the cluster size
def random_cluster(pr,cluster_path): # return fraction pr of the cluster samples randomly
    res = torch.load(cluster_path, map_location='cpu') 
    labels = res['labels'][0]
    num_cl = res['k']
    centers = res['centers'][0]
    if 'dino' not in cluster_path or 'imagenet' in cluster_path:
        feat = res['x_org']
    else:
        feat = res['x_org'][0]

    all_samples_idx = []
    for l in range(num_cl):
        cluster_sample_idx = torch.where(labels == l)[0]
        num_samples = int(len(cluster_sample_idx) * pr ) 
        pool = torch.rand(len(cluster_sample_idx))
        index = pool.topk(num_samples)[1]
        all_samples_idx.extend(cluster_sample_idx[index])
    return all_samples_idx

