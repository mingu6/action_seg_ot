import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
import asot

device = 'cpu'

vidname = '2020-04-19_14-03-25'  # desktop assembly
vidname = 'P36_webcam01_P36_salat'  # Breakfast (Salat)
dataset = 'da' if '2020' in vidname else 'salat'

## set ASOT parameters

if dataset == 'da':
    ub_weight = 0.09  # DA hparams
    eps = 0.01
    alpha = 0.6
    r = 0.02
else:
    ub_weight = 0.06  # BF (Salat) hparams
    eps = 0.01
    alpha = 0.6
    r = 0.04

affinity = torch.from_numpy(np.load(f'data/affinity/{vidname}.npy'))
affinity = affinity.unsqueeze(0)
matching_cost = 1. - affinity

## load ground truth

def process_mapping(x):
    i, nm = x.rstrip().split(' ')
    return nm, int(i)

action_mapping = dict(map(process_mapping, open(os.path.join('data', 'gt', f'mapping_{dataset}.txt'))))
gt = [line.rstrip() for line in open(os.path.join('data', 'gt', vidname))]
gt = torch.Tensor(list(map(lambda x: action_mapping[x], gt))).int().to(device)

## run ASOT
with torch.no_grad():
    _, N, K = matching_cost.shape
    soft_assign, _ = asot.segment_asot(matching_cost, eps=eps, alpha=alpha, radius=r, lambda_actions=ub_weight)
    segmentation = soft_assign.argmax(dim=2).squeeze(0)

fig = utils.plot_matrix(affinity.squeeze(0).cpu().T, gt=gt.cpu(), title='Raw affinity matrix')
plt.savefig('affinity.png')
plt.clf()
fig = utils.plot_matrix(soft_assign.squeeze(0).cpu().T, gt=gt.cpu(), title='ASOT solution (soft assignment)')
plt.savefig('soft_assign_asot.png')
plt.clf()
fig = utils.plot_segmentation_gt(gt, segmentation.squeeze(0), name=f'{vidname}')
plt.savefig('segmentation.png')
