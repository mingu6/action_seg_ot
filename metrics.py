
import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics.cluster import adjusted_rand_score

from torchmetrics import Metric


class ClusteringMetrics(Metric):
    def __init__(self, metric='nmi'):
        super().__init__()
        self.add_state("pred_labels", default=[], dist_reduce_fx="cat")
        self.add_state("gt_labels", default=[], dist_reduce_fx="cat")
        self.add_state("n_videos", default=torch.tensor([0.]), dist_reduce_fx="sum")
        self.metric_fn = score_fn_lookup[metric]

    def update(self, pred_labels, gt_labels, mask):
        self.pred_labels.extend(pred_labels.flatten()[mask.flatten()].tolist())
        self.gt_labels.extend(gt_labels.flatten()[mask.flatten()].tolist())
        self.n_videos += pred_labels.shape[0]

    def compute(self, exclude_cls=None):
        metric = self.metric_fn(np.array(self.pred_labels), np.array(self.gt_labels), self.n_videos, exclude_cls)
        return metric


def filter_exclusions(pred_labels, gt_labels, excl_cls):
    if excl_cls is None:
        return pred_labels, gt_labels
    mask = gt_labels != excl_cls
    return pred_labels[mask], gt_labels[mask]


def pred_to_gt_match(pred_labels, gt_labels):
    pred_uniq = np.unique(pred_labels)
    gt_uniq = np.unique(gt_labels)

    affinity_labels = np.zeros((len(pred_uniq), len(gt_uniq)))

    for pred_idx, pred_lab in enumerate(pred_uniq):
        for gt_idx, gt_lab in enumerate(gt_uniq):
            affinity_labels[pred_idx, gt_idx] = np.logical_and(
                pred_labels == pred_lab, gt_labels == gt_lab).sum()
    
    pred_idx_opt, gt_idx_opt = linear_sum_assignment(affinity_labels, maximize=True)
    pred_opt = pred_uniq[pred_idx_opt]
    gt_opt = gt_uniq[gt_idx_opt]
    return pred_opt, gt_opt


def eval_mof(pred_labels, gt_labels, n_videos, exclude_cls=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)

    true_pos_count = 0
    for pred_lab, gt_lab in zip(pred_opt, gt_opt):
        true_pos_count += np.logical_and(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()
    return true_pos_count / len(gt_labels_) , true_pos_count, len(gt_labels_)


def eval_iou(pred_labels, gt_labels, n_videos, exclude_cls=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)

    true_pos_count = 0
    union_count = 0
    for pred_lab, gt_lab in zip(pred_opt, gt_opt):
        true_pos_count += np.logical_and(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()
        union_count += np.logical_or(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()
    return true_pos_count / union_count, true_pos_count, union_count


def eval_f1(pred_labels, gt_labels, n_videos, exclude_cls=None, n_sample=15, n_exper=50, eps=1e-8, weird=False):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)
    n_actions = len(np.unique(gt_labels_))

    gt_segment_boundaries = np.where(gt_labels_[1:] - gt_labels_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries))

    tp_agg = 0.
    segments_count = 0

    for it in range(n_exper):
        for lo, up in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
            # if up - lo == 1:
            #     continue
            sample_idx = np.random.random_integers(lo, up, n_sample)
            gt_lab = gt_labels_[lo]
            if gt_lab in gt_opt:
                pred_lab = pred_opt[gt_opt == gt_lab]
                tp = (pred_labels_[sample_idx] == pred_lab).sum()
            else:
                tp = 0.  # never predicted this gt label, so no true positives
            if weird:
                tp_agg += tp / n_sample
            elif tp / n_sample > 0.5:
                tp_agg += 1
            if it == 0:
                segments_count += 1
    precision = tp_agg / (n_videos * n_actions * n_exper)
    recall = tp_agg / (segments_count * n_exper + eps)
    f1 = 2. * (precision * recall) / (precision + recall + eps)
    return f1, precision, recall, n_videos, segments_count


def eval_nmi(pred_labels, gt_labels, n_videos, exclude_cls=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    score = nmi_score(gt_labels_, pred_labels_)
    if type(score) is float:
        return score
    else:
        return score.item()


def eval_ari(pred_labels, gt_labels, n_videos, exclude_cls=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    return adjusted_rand_score(gt_labels_, pred_labels_)


def indep_eval_metrics(pred_labels_batch, gt_labels_batch, mask, metric, exclude_cls=None):
    """
    Evaluates each video sequence in a batch independently and aggregates results.
    """
    B = len(pred_labels_batch)
    eval_fn = score_fn_lookup[metric]
    val = 0.
    for b in range(B):
        score = eval_fn(pred_labels_batch[b][mask[b]].cpu().numpy(), gt_labels_batch[b][mask[b]].cpu().numpy(), 1, exclude_cls)
        if type(score) is float:
            val += score
        else:
            val += score[0]
    return val / B


score_fn_lookup = {'nmi': eval_nmi, 'ari': eval_ari, 'mof': eval_mof, 'f1': eval_f1, 'f1_w': lambda pl, gl, n, x: eval_f1(pl, gl, n, weird=True, exclude_cls=x), 'iou': eval_iou}