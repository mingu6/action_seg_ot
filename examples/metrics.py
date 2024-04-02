
import numpy as np

from scipy.optimize import linear_sum_assignment


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
