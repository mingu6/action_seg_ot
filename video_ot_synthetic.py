import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from seg_ot import segment_joint, segment_indep

torch.manual_seed(10)
np.random.seed(20)


def generate_clust(n_clust, feat_dim):
    return F.normalize(torch.randn(n_clust, feat_dim), dim=-1)


def generate_seq(n_seq, seq_len, feat_dim, n_clust, n_c_per_seq, clusters, noise_sigma):
    seqs = torch.zeros(n_seq, seq_len, feat_dim)
    # for each sequence, determine number of and which clusters are observed
    n_segments = torch.randint(low=n_c_per_seq[0], high=n_c_per_seq[1]+1, size=(n_seq,))
    # segment_idx = [torch.randperm(n_clust)[:n_seg] for n_seg in n_segments]
    segment_idx = [torch.randint(0, n_clust, (n_seg.item(),)) for n_seg in n_segments]
    # sample frame boundaries for each cluster
    splits = []
    for n_seg in n_segments:
        while True:
            split_idx = np.random.multinomial(seq_len, np.ones(n_seg) / n_seg, size=1)[0]
            if split_idx.min() > n_seq / (3 * n_seg):
                break
        split_idx = torch.from_numpy(np.concatenate(([0], split_idx)))
        splits.append(torch.cumsum(split_idx, 0))
    # generate data as noise applied to clusters
    for b in range(n_seq):
        for i in range(n_segments[b]):
            curr_clust = clusters[segment_idx[b][i]]
            start = splits[b][i]
            end = splits[b][i+1]
            seqs[b, start:end, :] = curr_clust[None, :] + torch.randn(end - start, feat_dim) * noise_sigma
    seqs = F.normalize(seqs, dim=2)
    return seqs, segment_idx, splits


def gt_segmentations(segment_idx, splits, seq_len, n_clust):
    n_seq = len(segment_idx)
    seqs = torch.zeros(n_seq, seq_len, n_clust)
    for b in range(n_seq):
        n_seg = len(segment_idx[b])
        for i in range(n_seg):
            start = splits[b][i]
            end = splits[b][i+1]
            seqs[b, start:end, segment_idx[b][i]] = 1.
    return seqs


# N = 20
# K = 8
# d = 5
# B = 2
# l, u = 3, 5
N = 40
K = 11
d = 32
B = 10
l, u = 5, 10


eps = 0.05
alpha = 0.5
# r = 3
r = 4
inter_cost = 0.0
proj_wt = 0.1
proj_type = 'kl'

n_iters = (100, 100)

clusters = generate_clust(K, d)
seqs, seg_idx, splits = generate_seq(B, N, d, K, (l, u), clusters, 0.6)
gt = gt_segmentations(seg_idx, splits, N, K)
mask = torch.ones(B, N).bool()

# output = segment_joint(seqs, clusters, mask, eps=eps, alpha=alpha, radius=r, inter_cost=inter_cost,
#                        proj=proj_type, proj_weight=proj_wt, n_iters=n_iters)
output = segment_indep(seqs, clusters, mask, eps=eps, alpha=alpha, radius=r, inter_cost=inter_cost,
                       proj=proj_type, proj_weight=proj_wt, n_iters=n_iters)

raw_sims = torch.einsum('bnd,cd->bcn', seqs, clusters)


fig, axs = plt.subplots(3, B, figsize=(B * 10, 3 * 5))

for b in range(B):
    axs[0, b].matshow(output[b].T)
    axs[1, b].matshow(gt[b].T)
    axs[2, b].matshow(raw_sims[b])

print(output.sum(dim=1).mean(dim=0))

fig.tight_layout()

plt.savefig('debug1.png')



