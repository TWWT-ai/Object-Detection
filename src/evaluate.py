import torch as th
import argparse
from pathlib import Path
 
from dataloader import get_dataLoaders
from model import HandGestureNet
from utils import compute_intersection_over_union
 
# Grid size, must match training
S = 7

#---------------------------------------------------
# Per-head metric helpers
#---------------------------------------------------

def extract_gt_box(det_target):
    """
    Inverse of encode_yolo_target for ONE image:
    [S, S, 5] grid -> the single (cx, cy, w, h) stored in the object cell.
    """
    object_mask = det_target[..., 4] > 0.5
    return det_target[object_mask][0, :4]


def extract_best_pred_box(det_pred, B=2):
    """
    From the raw head output of ONE image [S, S, B*5],
    return the (cx, cy, w, h) of the box with the highest confidence.
    Center format on purpose, so compute_intersection_over_union can
    compare it with the ground truth directly.
    """
    det_pred = det_pred.view(S, S, B, 5)
    conf = det_pred[..., 4]
    # argmax over the flattened S*S*B boxes, then unravel back to 3 indices
    flat_idx = conf.flatten().argmax()
    row = flat_idx // (S * B)
    col = (flat_idx % (S * B)) // B
    b = flat_idx % B
    return det_pred[row, col, b, :4]


def segmentation_scores(seg_logits, gt_mask, eps=1e-6):
    """
    Pixel IoU and Dice for one batch.
    seg_logits: [N, 1, H, W] raw logits (BCEWithLogitsLoss means the model
    outputs logits, so we must apply sigmoid HERE before thresholding).
    """
    pred_mask = (th.sigmoid(seg_logits) > 0.5).float()
    gt_mask = gt_mask.float()
 
    # Flatten each image, count overlaps per image, then sum over batch
    inter = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    pred_area = pred_mask.sum(dim=(1, 2, 3))
    gt_area = gt_mask.sum(dim=(1, 2, 3))
 
    iou = (inter / (pred_area + gt_area - inter + eps)).sum().item()
    dice = (2 * inter / (pred_area + gt_area + eps)).sum().item()
    return iou, dice