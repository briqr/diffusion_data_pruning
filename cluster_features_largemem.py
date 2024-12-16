import torch
from torch_kmeans import KMeans
import numpy as np
import os
from typing import NamedTuple
from torch import LongTensor, Tensor


def init_centers(x, n, d, k_max):
    gen = None
    rnd_idx = torch.multinomial(
        torch.empty((1, n), device=x.device, dtype=x.dtype).fill_(
            1 / n
        ),
        num_samples=k_max,
        replacement=False,
        generator=gen,
    )

    centers = x.gather(
            index=rnd_idx.view(1, -1)[:, :, None].expand(1, -1, d), dim=1
        ).view(1, 1, k_max, d)
    return centers[0,0]

def kmeans_cluster(data, k):
    num_iterations = 100

    centroids = init_centers(data.unsqueeze(0), data.shape[0], data.shape[1], k)
    for v in range(num_iterations):
        # Calculate distances from data points to centroids
        print('iteration %d' %v)
        d_size = 100000
        num_chunks = int(data.shape[0]/d_size+0.5)
        print('number of chunks is %d' %num_chunks)
        distances = []
        for s in range(num_chunks):
            print('distance for chunk %d'%s)
            current_d_chunk = data[s*d_size: (s+1)*d_size]
            current_distances = torch.cdist(current_d_chunk, centroids)
            print('current dista shape', current_distances.shape)
            distances.append(current_distances)
        distances = torch.cat(distances).float()
        print('distances', distances.shape)
    
        # Assign each data point to the closest centroid
        _, labels = torch.min(distances, dim=1)
    
        # Update centroids by taking the mean of data points assigned to each centroid
        for i in range(k):
            print('iteration %d, computing new centroids for cluster %d' %(v, i))
            if torch.sum(labels == i) > 0:
                centroids[i] = torch.mean(data[labels == i], dim=0)
    
    
    return centroids, labels

def main():
    num_clusters = [1000]
    feat_extractor = 'dino' #clip

    feat_path = 'pruning/datasets/imagenet/%s_features/train'%feat_extractor
    if feat_extractor == 'clip':
        f = []
        for s in range(0,7):
            current_f = torch.load(os.path.join(feat_path, 'imagenet_clip_features_%d.pth'%s))
            f += current_f
    else:
        f = torch.load(os.path.join(feat_path, 'imagenet_%s_features.pth')%feat_extractor)
    f = torch.from_numpy(np.asarray(f)).float() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    f = f.to(device)
    for num_cl in num_clusters:
        print('******num cl %d' %num_cl)
        centroids1, labels1 = kmeans_cluster(f, num_cl)
        inertia = 0
        for l in range(num_cl):
            cluster_sample_idx = torch.where(labels1 == l)[0]
            dist = torch.norm(f[cluster_sample_idx] - centroids1[l], dim=-1)
            inertia += dist.sum(dim=-1)
        inertia /= num_cl

        print('inertia', inertia.item())
        for l in range(num_cl):
            print('number of points in cluster %d' %l, (labels1==l).sum()  )

        result = dict({'labels': labels1.unsqueeze(0),  # type: ignore
            'centers': centroids1.unsqueeze(0),
            'inertia': inertia.item(),
            'x_org': f,
            'k': num_cl})
        torch.save(result, os.path.join(feat_path, 'cluster_vaeretrained_%s_%d.pth'%(feat_extractor, num_cl)))


       
if __name__ == "__main__":
    main()