"""YOLOv1 loss (Redmon et al. 2016, Eq. 3) + Dice/BCE segmentation loss.

The YOLOv1 loss is sum-squared error over five terms:
  1. coordinate loss (x, y)            — only for the box "responsible" for the object
  2. size loss (sqrt(w), sqrt(h))      — sqrt so small boxes are not dominated by large ones
  3. confidence loss for object cells  — target = IoU(pred, gt), per the paper
  4. confidence loss for empty cells   — weighted by LAMBDA_NOOBJ (most cells are empty)
  5. classification loss               — MSE on the one-hot class vector (paper choice;
                                         cross-entropy came later with YOLOv2/v3)
"responsible" = of the B boxes in the object's cell, the one with highest IoU vs the
ground truth.  This specialisation is the core YOLOv1 trick that lets B boxes per cell
learn different aspect ratios.
"""

import torch                       # tensor ops
import torch.nn as nn              # loss base class
import torch.nn.functional as F    # mse_loss / bce

from . import config               # S, B, C and the lambda weights


def _cell_to_xyxy(box, i, j):
    """Convert (x_cell, y_cell, w, h) at grid cell (i, j) to normalised corner coords.

    Needed twice: to compute the responsibility IoU inside the loss, and nowhere else —
    predictions and targets both live in cell-relative form during training.
    """
    cx = (j.float() + box[:, 0]) / config.S            # cell-relative x -> image-relative centre x
    cy = (i.float() + box[:, 1]) / config.S            # cell-relative y -> image-relative centre y
    w = box[:, 2]                                      # width already image-relative
    h = box[:, 3]                                      # height already image-relative
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)  # corners


def _iou_pairwise(a, b):
    """Element-wise IoU between two (M,4) corner-format tensors (row i vs row i)."""
    ix1 = torch.max(a[:, 0], b[:, 0])                  # intersection left edge
    iy1 = torch.max(a[:, 1], b[:, 1])                  # intersection top edge
    ix2 = torch.min(a[:, 2], b[:, 2])                  # intersection right edge
    iy2 = torch.min(a[:, 3], b[:, 3])                  # intersection bottom edge
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)      # clamp: disjoint -> 0
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-9)    # epsilon avoids division by zero


class YoloV1Loss(nn.Module):
    """Implements the 5-term YOLOv1 loss described above."""

    def forward(self, pred, target):
        N = pred.shape[0]                                                  # batch size (for averaging)
        boxes = pred[..., :config.B * 5].reshape(N, config.S, config.S, config.B, 5)  # split B boxes
        cls_pred = pred[..., config.B * 5:]                                # (N,S,S,C) class scores
        obj = target[..., 4] > 0.5                                         # (N,S,S) object indicator

        # ------- gather everything belonging to object cells (M = number of GT boxes) -------
        idx = obj.nonzero(as_tuple=False)                                  # (M,3): batch, row i, col j
        tb = target[..., :4][obj]                                          # (M,4) GT boxes (cell format)
        pb = boxes[obj]                                                    # (M,B,5) predicted boxes there
        gt_xyxy = _cell_to_xyxy(tb, idx[:, 1], idx[:, 2])                  # GT in corner format
        ious = torch.stack([_iou_pairwise(_cell_to_xyxy(pb[:, b, :4],     # IoU of each of the B boxes
                                                        idx[:, 1], idx[:, 2]), gt_xyxy)
                            for b in range(config.B)], dim=1)              # (M,B)
        best = ious.argmax(dim=1)                                          # responsible box index per GT
        rng = torch.arange(pb.shape[0], device=pred.device)                # row selector
        resp = pb[rng, best]                                               # (M,5) the responsible boxes
        best_iou = ious[rng, best].detach()                                # IoU used as confidence target
        #                                                                    (.detach(): it is a TARGET, we
        #                                                                    must not backprop through it)

        # ------- term 1+2: localisation (paper Eq. 3, first two sums) -------
        loss_xy = F.mse_loss(resp[:, 0:2], tb[:, 0:2], reduction='sum')    # centre offsets
        loss_wh = F.mse_loss(torch.sqrt(resp[:, 2:4].clamp(min=1e-6)),     # sqrt: equalises the gradient
                             torch.sqrt(tb[:, 2:4].clamp(min=1e-6)),       # scale of small vs large boxes
                             reduction='sum')

        # ------- term 3: confidence of responsible boxes (target = IoU, per paper) -------
        loss_obj = F.mse_loss(resp[:, 4], best_iou, reduction='sum')

        # ------- term 4: confidence of every box in EMPTY cells should be 0 -------
        noobj_conf = boxes[~obj][..., 4]                                   # (K,B) confidences in empty cells
        loss_noobj = (noobj_conf ** 2).sum()                               # MSE against target 0

        # ------- term 5: classification, MSE on one-hot (paper used SSE here too) -------
        loss_cls = F.mse_loss(cls_pred[obj], target[..., 5:][obj], reduction='sum')

        # ------- weighted sum, averaged over the batch (sum-style like the paper) -------
        total = (config.LAMBDA_COORD * (loss_xy + loss_wh)                 # localisation boosted x5
                 + loss_obj                                                # object confidence
                 + config.LAMBDA_NOOBJ * loss_noobj                        # empty cells damped x0.5
                 + loss_cls) / N                                           # class term, then batch mean
        return total


def seg_loss(logits, mask):
    """BCE + Dice loss for the segmentation head.

    Why BOTH?  BCE gives smooth per-pixel gradients everywhere; Dice directly optimises
    the overlap metric and is robust to the foreground/background imbalance (the hand
    covers a small fraction of the image, so plain BCE alone under-segments).
    """
    bce = F.binary_cross_entropy_with_logits(logits, mask)                 # per-pixel BCE (on logits:
    #                                                                        numerically stabler than
    #                                                                        sigmoid + plain BCE)
    prob = torch.sigmoid(logits)                                           # probabilities for Dice
    inter = (prob * mask).sum(dim=(1, 2, 3))                               # per-image intersection
    denom = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))              # per-image "union" (sums)
    dice = 1 - ((2 * inter + 1) / (denom + 1)).mean()                      # +1 smoothing avoids 0/0
    return bce + dice                                                      # equal mix works well here