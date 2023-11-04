import math
import numpy as np

import torch
import torch.nn.functional as F


def construct_Cv(N, r, device):
    frame_distance = torch.arange(N, device=device).unsqueeze(1)
    cost = (frame_distance - frame_distance.T).abs().float()
    cost[cost <= N * r] = 1.
    cost[cost > N * r] = 0.
    cost -= torch.eye(N, device=device)
    return cost / r


def construct_Ck(K, device):
    Ck = -torch.eye(K, device=device) + 1.
    return Ck


def aggregate_loss_tensor(Cv, Ck, Q):
    return Cv @ Q @ Ck.T


def proxdiv_tv(s, u, e, p, lam):
    lhs = torch.exp((lam - u) / e)
    rhs = torch.maximum(p / s, torch.exp(-(lam + u) / e))
    return torch.minimum(lhs, rhs)


def proxdiv_const(s, u, e, p, lam):
    return p / s


def proxdiv_kl(s, u, e, p, lam):
    return (p / s) ** (lam / (lam + e)) * torch.exp(-u / (lam + e))


def kl_mat(X, Y):
    return (X * torch.log(X / Y) - X + Y).sum()


def temporal_prior(n_frames, n_clusters, rho, device):
    temp_prior = torch.abs(torch.arange(n_frames)[:, None] / n_frames - torch.arange(n_clusters)[None, :] / n_clusters).to(device)
    return rho * temp_prior


def segment_asot(seq_features, clusters, mask, eps=0.05, alpha=0.3, radius=0.04, proj_type='const',
                 ub_weight=0.05, n_iters=(25, 15), stable_thres=7., temp_prior=None):
    B, N, _ = seq_features.shape
    n_c = clusters.shape[0]
    M = (1. - seq_features @ clusters.T.unsqueeze(0))
    nnz = mask.sum(dim=1)
    
    if temp_prior is not None:
        M = M + temp_prior

    dy = torch.ones((B, n_c, 1), device=M.device) / n_c
    dx = torch.ones((B, N, 1), device=M.device) / nnz[:, None, None]

    Q = dx * dy.transpose(1, 2)

    Cv = construct_Cv(N, radius, Q.device)
    Ck = construct_Ck(n_c, Q.device)

    if proj_type == 'kl':
        proxdiv_fn = proxdiv_kl
    elif proj_type == 'tv':
        proxdiv_fn = proxdiv_tv
    elif proj_type == 'const':
        proxdiv_fn = proxdiv_const
    else:
        raise ValueError('Invalid proxdiv function for scaling iterations')

    for it in range(n_iters[0]):
        CT = aggregate_loss_tensor(Cv, Ck, Q)
        cost = ((1. - alpha) * M + alpha * CT)
        cost = cost / cost.max()
        
        K = torch.exp(-cost / eps) * mask.unsqueeze(2)
        b = torch.ones((B, n_c, 1), device=Q.device)
        u = torch.zeros((B, N, 1), device=Q.device)
        v = torch.zeros((n_c, 1), device=Q.device)

        for i in range(n_iters[1]):
            a = proxdiv_const(K @ b, u, eps, dx, ub_weight)
            a = torch.nan_to_num(a, posinf=0., neginf=0.)
            b = proxdiv_fn(K.transpose(1, 2) @ a, v, eps, dy, ub_weight)
            b = torch.nan_to_num(b, posinf=0., neginf=0.)
            if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
                if i != n_iters[1] - 1:
                    u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                    v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                    K = torch.exp((u + v.transpose(1, 2) - cost) / eps) * mask.unsqueeze(2)
                    b = torch.ones_like(b)
        Q = a * K * b.transpose(1, 2)
    Q = Q * nnz[:, None, None]  # rescale so marginals per frame sum to 1
    return Q  


# below used for debugging

def obj(Q, Cv, Ck):
    return (aggregate_loss_tensor(Cv, Ck, Q) * Q).sum(dim=[1, 2])

def ot_objective(Q, mask, radius, partial_wt, eps):
    B, N, n_c = Q.shape

    nnz = mask.sum(dim=1)

    Cv = construct_Cv(N, radius, Q.device)
    Ck = construct_Ck(n_c, Q.device)

    dy = torch.ones((B, n_c, 1), device=M.device) / n_c

    linear_term = (aggregate_loss_tensor(Cv, Ck, Q) * Q).sum(dim=[1, 2])
    return linear_term + partial_wt * reg_partial(Q, dy) - eps * entropy(Q)

def reg_partial(Q, dy):
    marginal = Q.sum(dim=1)
    return (marginal * torch.log(marginal / dy.transpose(1, 2)) - marginal + dy.transpose(1, 2)).sum(dim=[1, 2])

def entropy(Q):
    return -(Q * (torch.log(Q) - 1.)).sum(dim=[1, 2])

def kld_pair(A, B):
    return (A * torch.log(A / B) - A + B).sum()

def kld(A):
    return (A * torch.log(A) - A + 1.).sum()

