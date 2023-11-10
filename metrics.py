
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

    def compute(self, exclude_cls=None, pred_to_gt=None):
        metric, pred_to_gt = self.metric_fn(np.array(self.pred_labels), np.array(self.gt_labels), self.n_videos, exclude_cls, pred_to_gt)
        return metric, pred_to_gt


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


def eval_mof(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt, gt_opt))
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())

    true_pos_count = 0
    for pred_lab, gt_lab in zip(pred_opt, gt_opt):
        true_pos_count += np.logical_and(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()
    return true_pos_count / len(gt_labels_) , pred_to_gt


def eval_miou(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt, gt_opt))
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())

    class_tp = []
    class_union = []

    for pred_lab, gt_lab in zip(pred_opt, gt_opt):
        class_tp += [np.logical_and(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()]
        class_union += [np.logical_or(pred_labels_ == pred_lab, gt_labels_ == gt_lab).sum()]

    mean_iou = sum([tp / un for tp, un in zip(class_tp, class_union)]) / len(np.unique(gt_labels_))
    return mean_iou, pred_to_gt


def eval_f1(pred_labels, gt_labels, n_videos, exclude_cls=None, pred_to_gt=None, n_sample=15, n_exper=50, eps=1e-8):
    pred_labels_, gt_labels_ = filter_exclusions(pred_labels, gt_labels, exclude_cls)
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_labels_, gt_labels_)
        pred_to_gt = dict(zip(pred_opt, gt_opt))
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())
        pred_opt, gt_opt  = np.array(pred_opt), np.array(gt_opt)
    n_actions = len(np.unique(gt_labels_))

    gt_segment_boundaries = np.where(gt_labels_[1:] - gt_labels_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_labels_)-1]))

    tp_agg = 0.
    segments_count = 0

    for it in range(n_exper):
        for lo, up in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
            sample_idx = np.random.random_integers(lo, up, n_sample)
            gt_lab = gt_labels_[lo]
            if gt_lab in gt_opt:
                pred_lab = pred_opt[gt_opt == gt_lab]
                tp = (pred_labels_[sample_idx] == pred_lab).sum()
            else:
                tp = 0.  # never predicted this gt label, so no true positives
            if tp / n_sample > 0.5:
                tp_agg += 1
            if it == 0:
                segments_count += 1
    precision = tp_agg / (n_videos * n_actions * n_exper)
    recall = tp_agg / (segments_count * n_exper + eps)
    f1 = 2. * (precision * recall) / (precision + recall + eps)
    return f1, pred_to_gt


def indep_eval_metrics(pred_labels_batch, gt_labels_batch, mask, metrics=['mof', 'f1', 'miou'], exclude_cls=None, pred_to_gt=None):
    """
    Evaluates each video sequence in a batch independently and aggregates results. Handles multiple metrics at once
    """
    B = len(pred_labels_batch)

    values = {metric: 0. for metric in metrics}

    for b in range(B):
        p2gt_local = None if pred_to_gt is None else pred_to_gt
        for metric in metrics:
            eval_fn = score_fn_lookup[metric]
            score, p2gt_local = eval_fn(pred_labels_batch[b][mask[b]].cpu().numpy(), gt_labels_batch[b][mask[b]].cpu().numpy(), 1, exclude_cls, p2gt_local)
            values[metric] += score / B
    return values


score_fn_lookup = {'mof': eval_mof, 'f1': eval_f1, 'miou': eval_miou}