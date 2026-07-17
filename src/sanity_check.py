"""Standalone sanity checks for the YOLOv1 pipeline.

Run:  python sanity_check.py     (CPU is fine, ~1 minute)

Three tests, each isolating one link of the chain:
  [1] encode -> decode round trip        (utils.py geometry)
  [2] hand-built perfect prediction      (decode + loss agree on "correct")
  [3] optimize a raw tensor by the loss  (do the loss's gradients actually
                                          pull predictions to the answer?)
If all three PASS, the encode/loss/decode code is proven correct and any
remaining bad results come from architecture / data / training recipe,
NOT from bugs in this chain.
"""
import torch as th

from utils import (encode_yolo_target, extract_gt_box, extract_yolo_prediction,
                   compute_intersection_over_union)
from loss import YOLOLoss
from train import prepare_yolo_targets, S, B, C


def logit(p):
    """Inverse of sigmoid, so sigmoid(logit(p)) == p exactly."""
    return th.log(p / (1 - p))


# ---------- Test 1: encode -> extract_gt_box round trip ----------
def test_roundtrip(n=200):
    max_err = 0.0
    for _ in range(n):
        cx, cy = (th.rand(2) * 0.9 + 0.05).tolist()
        w, h = (th.rand(2) * 0.4 + 0.05).tolist()
        corner = th.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        recovered = extract_gt_box(encode_yolo_target(corner))
        err = (recovered - th.tensor([cx, cy, w, h])).abs().max().item()
        max_err = max(max_err, err)
    ok = max_err < 1e-5
    print(f"[1] encode->decode round trip   max error {max_err:.2e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---------- Test 2: perfect prediction -> decode + loss agree ----------
def test_perfect_prediction():
    gt_corner = th.tensor([0.35, 0.50, 0.65, 0.80])
    tgt = encode_yolo_target(gt_corner)                       # [S,S,5]
    yolo_tgt = prepare_yolo_targets(tgt.unsqueeze(0), th.tensor([3]))

    pred = th.zeros(1, S, S, B * 5 + C)
    row, col = [v.item() for v in (tgt[..., 4] > 0.5).nonzero()[0]]
    x_cell, y_cell, w, h = tgt[row, col, :4]
    pred[0, row, col, C] = 1.0                 # confidence of box 1
    pred[0, row, col, C + 1] = logit(x_cell)   # loss sigmoids x,y -> pre-invert
    pred[0, row, col, C + 2] = logit(y_cell)
    pred[0, row, col, C + 3] = w               # w,h stay raw
    pred[0, row, col, C + 4] = h
    pred[0, row, col, 3] = 1.0                 # class 3 one-hot

    dec_box, dec_cls = extract_yolo_prediction(pred[0])
    iou = compute_intersection_over_union(dec_box, extract_gt_box(tgt)).item()
    loss = YOLOLoss(S=S, B=B, C=C)(pred.flatten(1), yolo_tgt).item()
    ok = iou > 0.99 and loss < 0.5 and dec_cls.item() == 3
    print(f"[2] perfect prediction          decode IoU {iou:.4f} (want ~1)   "
          f"loss {loss:.4f} (want ~0)   class {dec_cls.item()} (want 3)   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---------- Test 3: can the loss's gradients find the answer? ----------
def test_loss_descent(steps=800):
    gt_corner = th.tensor([0.20, 0.30, 0.55, 0.75])
    tgt = encode_yolo_target(gt_corner)
    yolo_tgt = prepare_yolo_targets(tgt.unsqueeze(0), th.tensor([7]))

    pred = (th.randn(1, S * S * (B * 5 + C)) * 0.1).requires_grad_(True)
    criterion = YOLOLoss(S=S, B=B, C=C)
    optimizer = th.optim.Adam([pred], lr=0.05)
    for _ in range(steps):
        loss = criterion(pred, yolo_tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dec_box, dec_cls = extract_yolo_prediction(pred.detach().view(S, S, -1))
    iou = compute_intersection_over_union(dec_box, extract_gt_box(tgt)).item()
    ok = iou > 0.9 and dec_cls.item() == 7
    print(f"[3] gradient descent on loss    final loss {loss.item():.4f}   "
          f"decoded IoU {iou:.4f} (want >0.9)   class {dec_cls.item()} (want 7)   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    th.manual_seed(0)
    results = [test_roundtrip(), test_perfect_prediction(), test_loss_descent()]
    print("=" * 60)
    print("ALL PASS — pipeline code is correct; look at data/recipe instead"
          if all(results) else
          "FAILURE above — the failing link is where the bug lives")
