import torch
import torch.nn.functional as F


# Fast implmentation of Cv @ T @ Ck.T for GW component of ASOT with O(NK) complexity

def construct_Cv_filter(N, r, device):
    abs_r = int(N * r)
    weights = torch.ones(2 * abs_r + 1, device=device) / r
    weights[abs_r] = 0.
    return weights[None, None, :]


def mult_Cv(Cv_weights, X):
    B, N, K = X.shape
    Y_flat = F.conv1d(X.transpose(1, 2).reshape(-1, 1, N), Cv_weights, padding='same')
    return Y_flat.reshape(B, K, N).transpose(1, 2)


# ASOT objective function gradients for mirro descent solver

def grad_fgw(T, cost_matrix, alpha, Cv):
    T_Ck = T.sum(dim=2, keepdim=True) - T
    return alpha * mult_Cv(Cv, T_Ck) + (1. - alpha) * cost_matrix

def grad_kld(T, p, lambd, axis):
    # p is marginal, dim is marginal axes
    marg = T.sum(dim=axis, keepdim=True)
    return lambd * (torch.log(marg / p + 1e-12) + 1.)

def grad_entropy(T, eps):
    return - torch.log(T + 1e-12) * eps


# Sinkhorn projections for balanced ASOT (balanced assignment for frames AND actions)

def project_to_polytope_KL(cost_matrix, mask, eps, dx, dy, n_iters=10, stable_thres=7.):
    # runs sinkhorn algorithm on dual potentials w/log domain stabilization
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    dual_pot = torch.exp(-cost_matrix / eps) * mask.unsqueeze(2)
    dual_pot = dual_pot / dual_pot.max()
    b = torch.ones((B, K, 1), device=dev)
    u = torch.zeros((B, N, 1), device=dev)
    v = torch.zeros((K, 1), device=dev)

    for i in range(n_iters):
        a = dx / (dual_pot @ b)
        a = torch.nan_to_num(a, posinf=0., neginf=0.)
        b = dy / (dual_pot.transpose(1, 2) @ a)
        b = torch.nan_to_num(b, posinf=0., neginf=0.)
        if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
            if i != n_iters - 1:
                u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                dual_pot = torch.exp((u + v.transpose(1, 2) - cost_matrix) / eps) * mask.unsqueeze(2)
                b = torch.ones_like(b)
    T = a * dual_pot * b.transpose(1, 2)
    return T


# ASOT objective function evaluation

def kld(a, b, eps=1e-10):
    return (a * torch.log(a / b + eps)).sum(dim=1)


def entropy(T, eps=1e-10):
    return (-T * torch.log(T + eps) + T).sum(dim=(1, 2))


def asot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                   lambda_frames, lambda_actions, mask=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
        
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)
    T_mask = T * mask.unsqueeze(2)
    
    # FGW stuff
    Cv = construct_Cv_filter(N, radius, dev)
    fgw_obj = (grad_fgw(T_mask, cost_matrix, alpha, Cv) * T_mask).sum(dim=(1, 2))
    
    # Unbalanced stuff
    dy = torch.ones((B, K), device=dev) / K
    dx = torch.ones((B, N), device=dev) / nnz[:, None]
    
    frames_marg = T_mask.sum(dim=2)
    frames_ub_penalty = kld(frames_marg, dx) * lambda_frames
    actions_marg = T_mask.sum(dim=1)
    actions_ub_penalty = kld(actions_marg, dy) * lambda_actions
    
    ub = torch.zeros(B, device=dev)
    if ub_frames:
        ub += frames_ub_penalty
    if ub_actions:
        ub += actions_ub_penalty
    
    # entropy reg
    entr = -eps * entropy(T)
    
    # objective
    obj = 0.5 * fgw_obj + ub + entr
    
    return obj


# ASOT solver

def segment_asot(cost_matrix, mask=None, eps=0.07, alpha=0.3, radius=0.04, ub_frames=False,
                 ub_actions=True, lambda_frames=0.1, lambda_actions=0.05, n_iters=(25, 1),
                 stable_thres=7., step_size=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)

    dy = torch.ones((B, K, 1), device=dev) / K
    dx = torch.ones((B, N, 1), device=dev) / nnz[:, None, None]

    T = dx * dy.transpose(1, 2)
    T = T * mask.unsqueeze(2)
    
    Cv = construct_Cv_filter(N, radius, dev)
    
    obj_trace = []
    it = 0

    while True:
        with torch.no_grad():
            obj = asot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                                lambda_frames, lambda_actions, mask=mask)
        obj_trace.append(obj)
        
        if it >= n_iters[0]:
            break
        
        # gradient of objective function required for mirror descent step
        fgw_cost_matrix = grad_fgw(T, cost_matrix, alpha, Cv)
        grad_obj = fgw_cost_matrix - grad_entropy(T, eps)
        if ub_frames:
            grad_obj += grad_kld(T, dx, lambda_frames, 2)
        if ub_actions:
            grad_obj += grad_kld(T, dy.transpose(1, 2), lambda_actions, 1)
        
        # automatically calibrate stepsize by rescaling based on observed gradient
        if it == 0 and step_size is None:
            step_size = 4. / grad_obj.max().item()

        # update step - note, no projection required if both sides are unbalanced
        T = T * torch.exp(-step_size * grad_obj)
        
        if not ub_frames and not ub_actions:
            T = project_to_polytope_KL(fgw_cost_matrix, mask, eps, dx, dy,
                                       n_iters=n_iters[1], stable_thres=stable_thres)
        elif not ub_frames:
            T /= T.sum(dim=2, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dx
        elif not ub_actions:
            T /= T.sum(dim=1, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dy.transpose(1, 2)
        
        it += 1
    
    T = T * nnz[:, None, None]  # rescale so marginals per frame sum to 1
    obj_trace = torch.cat(obj_trace)
    return T, obj_trace


def temporal_prior(n_frames, n_clusters, rho, device):
    temp_prior = torch.abs(torch.arange(n_frames)[:, None] / n_frames - torch.arange(n_clusters)[None, :] / n_clusters).to(device)
    return rho * temp_prior