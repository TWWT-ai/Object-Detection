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
    dice_coef = (2 * inter / (pred_area + gt_area + eps)).sum().item()
    return iou, dice_coef

#---------------------------------------------------
# Evaluation loop
#---------------------------------------------------
 
@th.no_grad()
def evaluate(model, loader, device, n_classes=10, iou_thresh=0.5):
    model.eval()
 
    # Accumulators
    confusion = th.zeros(n_classes, n_classes, dtype=th.long)   # rows = truth, cols = prediction
    sum_seg_iou, sum_seg_dice = 0.0, 0.0
    sum_det_iou, det_hits = 0.0, 0
    total = 0
 
    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        det_pred, seg_pred, cls_pred = model(images)
        n = images.size(0)
        total += n
 
        # ---- Classification: fill the confusion matrix ----
        pred_label = cls_pred.argmax(dim=1)
        for t, p in zip(targets["Label"], pred_label):
            confusion[t.item(), p.item()] += 1
 
        # ---- Segmentation: pixel IoU + Dice ----
        batch_iou, batch_dice = segmentation_scores(seg_pred, targets["Mask"])
        sum_seg_iou += batch_iou
        sum_seg_dice += batch_dice
 
        # ---- Detection: best box vs ground truth box, one pair per image ----
        for i in range(n):
            gt_box = extract_gt_box(targets["det"][i])
            pred_box = extract_best_pred_box(det_pred[i])
            iou = compute_intersection_over_union(pred_box, gt_box).item()
            sum_det_iou += iou
            if iou >= iou_thresh:
                det_hits += 1
 
    # ---- Aggregate ----
    per_class_correct = confusion.diag().float()
    per_class_total = confusion.sum(dim=1).float()
 
    metrics = {
        "cls_acc": (per_class_correct.sum() / total).item(),
        "cls_acc_per_class": (per_class_correct / per_class_total.clamp(min=1)).tolist(),
        "confusion": confusion,
        "seg_iou": sum_seg_iou / total,
        "seg_dice": sum_seg_dice / total,
        "det_mean_iou": sum_det_iou / total,
        "det_acc": det_hits / total,          # fraction of images with IoU >= iou_thresh
        "n_samples": total,
    }
    return metrics


def print_report(m, iou_thresh=0.5):
    print("=" * 52)
    print(f"Evaluation on {m['n_samples']} validation images")
    print("=" * 52)
    print(f"[Classification] accuracy       : {m['cls_acc']:.4f}")
    for c, acc in enumerate(m["cls_acc_per_class"]):
        print(f"                 G{c + 1:02d} accuracy   : {acc:.4f}")
    print(f"[Segmentation]   pixel IoU      : {m['seg_iou']:.4f}")
    print(f"                 Dice           : {m['seg_dice']:.4f}")
    print(f"[Detection]      mean IoU       : {m['det_mean_iou']:.4f}")
    print(f"                 acc @ IoU>={iou_thresh}: {m['det_acc']:.4f}")
    print("\nConfusion matrix (rows = truth G01..G10, cols = prediction):")
    header = "     " + " ".join(f"P{c + 1:02d}" for c in range(m["confusion"].size(1)))
    print(header)
    for r in range(m["confusion"].size(0)):
        row = " ".join(f"{v:3d}" for v in m["confusion"][r].tolist())
        print(f"G{r + 1:02d}  {row}")