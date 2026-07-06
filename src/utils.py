import torch as th


def compute_intersection_over_union(box1, box2):
    # Computing left and right boundary for both boxes (cx, cy, w, h)
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2         # left bound
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2         # right bound
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    # Calculating intersection for height and width
    inter_w = (th.min(b1_x2, b2_x2) - th.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (th.min(b1_y2, b2_y2) - th.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    # Calculating union
    union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - inter
    return inter / (union + 1e-6)
