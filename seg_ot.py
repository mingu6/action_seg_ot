import numpy as np

import torch
import torch.nn.functional as F


def construct_Cv_joint(N, r, B, c, device):
    intra_cost = construct_Cv_indep(N, r, device)
    full_cost = torch.full((B * N, B * N), c, dtype=torch.float, device=device)
    for b in range(B):
        full_cost[b*N:(b+1)*N, b*N:(b+1)*N] = intra_cost
    return full_cost


# def construct_Cv_indep(N, r, device):
#     frame_distance = torch.arange(N, device=device).unsqueeze(1)
#     cost = (frame_distance.T - frame_distance).float()
#     cost[(cost <= r) * (cost > 0.)] = 1.
#     cost[cost < 0.] = 0.
#     cost[cost > r] = 0.

#     cost /= (cost.sum(dim=1, keepdim=True) + 1e-8)

#     return cost

def construct_Cv_indep(N, r, device):
    frame_distance = torch.arange(N, device=device).unsqueeze(1)
    cost = (frame_distance - frame_distance.T).abs().float()
    cost[cost <= r] = 1.# / r
    cost[cost > r] = 0.
    cost -= torch.eye(N, device=device)
    # cost /= (cost.sum(dim=1)[:, None] + 1e-8)
    return cost


def construct_Ck(K, beta, device):
    Ck = -torch.eye(K, device=device) + 1.

    # Ck_l = torch.tril(Ck)
    # Ck_u = Ck_l.T * beta
    # Ck = Ck_l + Ck_u
    # Ck += torch.eye(K, device=device) * 0.
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


def segment_joint(seq_features, clusters, mask, eps=0.05, alpha=0.3, radius=10, inter_cost=0.1, sigma=5., rho=0.1,
                  proj='const', proj_weight=0.1, n_iters=(10, 10), stable_thres=7.):
    B, N, _ = seq_features.shape
    n_c = clusters.shape[0]
    M = (1. - seq_features @ clusters.T.unsqueeze(0))
    M = M.reshape(B * N, n_c)

    if proj == 'kl':
        proxdiv_fn = proxdiv_kl
    elif proj == 'tv':
        proxdiv_fn = proxdiv_tv
    elif proj == 'const':
        proxdiv_fn = proxdiv_const
    else:
        raise ValueError('Invalid proxdiv function for scaling iterations')
    
    temp_prior = torch.abs(torch.arange(N)[:, None] / N - torch.arange(n_c)[None, :] / n_c) / np.sqrt(1. / N ** 2 + 1 / n_c ** 2)
    temp_prior = torch.exp(-temp_prior / (sigma))
    temp_prior /= temp_prior.sum(dim=1, keepdim=True)
    M -= rho * torch.log(temp_prior.to(M.device)).repeat(B, 1)

    Q = torch.exp(-M / eps)
    Q = Q / Q.sum(dim=1, keepdim=True)

    Cv = construct_Cv_joint(N, radius, B, inter_cost, Q.device)
    Ck = construct_Ck(n_c, 0.1, Q.device)
    nnz = mask.sum()

    for it in range(n_iters[0]):
        CT = aggregate_loss_tensor(Cv, Ck, Q)

        cost = ((1. - alpha) * M + alpha * CT)
        cost /= cost.max()
        
        dx = torch.full((B * N, 1), 1. / nnz, device=Q.device)
        dy = torch.full((n_c, 1), 1. / n_c, device=Q.device)
        K = torch.exp(-cost / eps) * mask.reshape(-1).unsqueeze(1)
        b = torch.ones((n_c, 1), device=Q.device)
        u = torch.zeros((B * N, 1), device=Q.device)
        v = torch.zeros((n_c, 1), device=Q.device)
        
        for i in range(n_iters[1]):
            a = proxdiv_const(K @ (b * dy), u, eps, dx, proj_weight)
            a = torch.nan_to_num(a, posinf=0., neginf=0.)
            b = proxdiv_fn(K.T @ (a * dx), v, eps, dy, proj_weight)
            b = torch.nan_to_num(b, posinf=0., neginf=0.)
            if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
                if i != n_iters[1] - 1:
                    u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                    v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                    K = torch.exp((u + v.T - cost) / eps) * mask.reshape(-1).unsqueeze(1)
                    b = torch.ones_like(b)
            
        Q = a * K * b.T
    Q = Q.reshape(B, N, n_c)
    Q *= nnz / Q.sum()
    return Q


def segment_indep(seq_features, clusters, mask, eps=0.05, alpha=0.3, radius=10, sigma=5., 
                  proj='const', proj_weight=0.1, n_iters=(3, 2), stable_thres=7., rho=0.1):
    B, N, _ = seq_features.shape
    n_c = clusters.shape[0]
    M = (1. - seq_features @ clusters.T.unsqueeze(0))
    # Mp = torch.exp(-M)# / 0.5)
    # Mp /= Mp.sum(dim=2, keepdim=True)
    # M = -torch.log(Mp)
    nnz = mask.sum(dim=1)

    # Q = torch.exp(-M)
    # Q = Q / Q.sum(dim=2, keepdim=True)
    

    temp_prior = torch.abs(torch.arange(N)[:, None] / N - torch.arange(n_c)[None, :] / n_c).to(M.device)
    M += rho * temp_prior

    Q = torch.exp(-M)
    Q = Q / Q.sum(dim=2, keepdim=True)
    # Q = torch.ones_like(M) / n_c

    Cv = construct_Cv_indep(N, radius, Q.device)
    Ck = construct_Ck(n_c, 1., Q.device)

    dy = (nnz / n_c)[:, None, None].repeat(1, n_c, 1)

    if proj == 'kl':
        proxdiv_fn = proxdiv_kl
    elif proj == 'tv':
        proxdiv_fn = proxdiv_tv
    elif proj == 'const':
        proxdiv_fn = proxdiv_const
    else:
        raise ValueError('Invalid proxdiv function for scaling iterations')

    for it in range(n_iters[0]):
        CT = aggregate_loss_tensor(Cv, Ck, Q)
        cost = ((1. - alpha) * M + alpha * CT)
        cost = cost / cost.max()
        
        dx = torch.ones((B, N, 1), device=Q.device)
        dy = (nnz / n_c)[:, None, None].repeat(1, n_c, 1)
        K = torch.exp(-cost / eps) * mask.unsqueeze(2)
        b = torch.ones((B, n_c, 1), device=Q.device)
        u = torch.zeros((B, N, 1), device=Q.device)
        v = torch.zeros((n_c, 1), device=Q.device)

        for i in range(n_iters[1]):
            a = proxdiv_const(K @ b, u, eps, dx, proj_weight)
            a = torch.nan_to_num(a, posinf=0., neginf=0.)
            b = proxdiv_fn(K.transpose(1, 2) @ a, v, eps, dy, proj_weight)
            b = torch.nan_to_num(b, posinf=0., neginf=0.)
            if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
                if i != n_iters[1] - 1:
                    u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                    v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                    K = torch.exp((u + v.transpose(1, 2) - cost) / eps) * mask.unsqueeze(2)
                    b = torch.ones_like(b)
        Q = a * K * b.transpose(1, 2)
    return Q  

def obj(Q, Cv, Ck):
    return (aggregate_loss_tensor(Cv, Ck, Q) * Q).sum(dim=[1, 2])

# def obj1(Q, Cv, Ck):
#     temp = [0., 0.]
#     from tqdm import tqdm
#     Qc = Q.detach().cpu()
#     Cvc = Cv.detach().cpu()
#     Ckc = Ck.detach().cpu()
#     for b in range(Q.shape[0]):
#         for i in tqdm(range(Q.shape[1])):
#             for j in range(Q.shape[1]):
#                 for k in range(Q.shape[2]):
#                     for l in range(Q.shape[2]):
#                         temp[b] += Qc[b, i, k] * Qc[b, j, l] * Cvc[i, j] * Ckc[k, l]
#     return temp

def ot_objective(Q, mask, radius, partial_wt, eps):
    B, N, n_c = Q.shape

    nnz = mask.sum(dim=1)

    Cv = construct_Cv_indep(N, radius, Q.device)
    Ck = construct_Ck(n_c, 1., Q.device)

    dy = (nnz / n_c)[:, None, None].repeat(1, n_c, 1)

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

