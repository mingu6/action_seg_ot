import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
import wandb

from metrics import pred_to_gt_match, filter_exclusions


def plot_segmentation(gt, pred, mask, gt_uniq=None, pred_to_gt=None, exclude_cls=None, name=''):
    colors = {}
    cmap = plt.get_cmap('tab20')

    pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy(), exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())
    for pr_lab, gt_lab in zip(pred_opt, gt_opt):
        pred_[pred_ == pr_lab] = gt_lab
    n_frames = len(pred_)

    # add colors for predictions which do not match to a gt class

    if gt_uniq is None:
        gt_uniq = np.unique(gt_.cpu().numpy())
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / len(gt_uniq))

    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel('GT', fontsize=30, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    gt_segment_boundaries = np.where(gt_[1:] - gt_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

    for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
        label = gt_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)

    # plot predicted segmentation after matching to gt labels w/Hungarian

    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel('Ours', fontsize=30, rotation=0, labelpad=60, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred_[1:] - pred_[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)

    fig.tight_layout()
    return fig


def plot_pairwise_frame_similarities(fname, idx, features, gt, pl_global_step):
    # pairwise intra-video cosine distances
    gt_change = np.where((np.diff(gt.cpu().numpy()) != 0))[0] + 1
    fdists = squareform(pdist(features.cpu().numpy(), 'cosine'))
    fdists = np.nan_to_num(fdists)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot1 = ax.matshow(fdists)
    for ch in gt_change:
        ax.axvline(ch, color='red')
    plt.colorbar(plot1, ax=ax)
    ax.set_title(fname)
    ax.set_xlabel('Frame idx')
    ax.set_ylabel('Frame idx')
    wandb.log({f"val_pairwise_{idx}": fig, "trainer/global_step": pl_global_step})
    plt.close()


def plot_frame_cluster_similarities(fname, idx, codes, gt, pl_global_step):
    gt_change = np.where((np.diff(gt.cpu().numpy()) != 0))[0] + 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot1 = ax.matshow(codes.cpu().numpy().T)
    for ch in gt_change:
        ax.axvline(ch, color='red')
    ax.set_aspect('auto')
    plt.colorbar(plot1, ax=ax)
    ax.set_title(fname)
    ax.set_xlabel('Frame idx')
    ax.set_ylabel('Cluster idx')
    wandb.log({f"val_P_{idx}": fig, "trainer/global_step": pl_global_step}) 
    plt.close()


def plot_pseudo_labels(fname, idx, pseudo_labels, gt, pl_global_step):
    gt_change = np.where((np.diff(gt.cpu().numpy()) != 0))[0] + 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot1 = ax.matshow(pseudo_labels.cpu().numpy().T)
    for ch in gt_change:
        ax.axvline(ch, color='red')
    ax.set_aspect('auto')
    plt.colorbar(plot1, ax=ax)
    ax.set_title(fname[0])
    ax.set_xlabel('Frame idx')
    ax.set_ylabel('Cluster idx')
    wandb.log({f"val_OT_PL_{idx}": fig, "trainer/global_step": pl_global_step}) 
    plt.close()


def plot_predictions(fname, idx, segmentation, gt, pl_global_step):
    gt_change = np.where((np.diff(gt.cpu().numpy()) != 0))[0] + 1
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot1 = ax.matshow(segmentation.cpu().numpy().T)
    for ch in gt_change:
        ax.axvline(ch, color='red')
    ax.set_aspect('auto')
    plt.colorbar(plot1, ax=ax)
    ax.set_title(fname[0])
    ax.set_xlabel('Frame idx')
    ax.set_ylabel('Cluster idx')
    wandb.log({f"val_OT_pred_{idx}": fig, "trainer/global_step": pl_global_step})
    plt.close()
