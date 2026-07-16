import torch as th

S = 7

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


def encode_yolo_target(boundary_box, s=S):
    x_min, y_min, x_max, y_max = [float(v) for v in boundary_box]
 
    # Corner format -> centre format
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min

    # Which grid cell does the centre fall into?
    # cx * s is in [0, s); int() floors it; min() guards the cx == 1.0 edge case
    col = min(int(cx * s), s - 1)     # column index <- x direction
    row = min(int(cy * s), s - 1)     # row index    <- y direction
    
    target = th.zeros(s, s, 5)
    target[row, col] = th.tensor([cx, cy, w, h, 1.0])
    return target


def extract_yolo_prediction(pred, s=S, B=2, C=20):
    """
    Decoder for the YOLOv1 layout: each cell holds
    [C class scores, conf_1, box_1(cx,cy,w,h), conf_2, box_2(cx,cy,w,h)].

    pred: flat [S*S*(B*5+C)] or [S,S,B*5+C] for ONE image.
    Returns (best box tensor [cx,cy,w,h], predicted class index).
    Only the first 10 class slots are real gestures (C=20 pads with zeros).
    """
    pred = pred.view(s, s, B * 5 + C)

    # Confidence of every box: index C for box 0, C+5 for box 1
    confs = th.stack([pred[..., C], pred[..., C + 5]], dim=-1)   # [S, S, B]
    flat_idx = confs.flatten().argmax()
    row = flat_idx // (s * B)
    col = (flat_idx % (s * B)) // B
    b = flat_idx % B

    box = pred[row, col, C + 1 + b * 5 : C + 5 + b * 5]
    cls_idx = pred[row, col, :10].argmax()      # gesture lives in slots 0-9
    return box, cls_idx


def decode_predictions(prediction, B=2, conf_thresh=0.25, s=S):
    """
    Inverse of the model output: [S, S, B*5] (one image, after permute)
    -> list of (x_min, y_min, x_max, y_max, conf), all in [0, 1].
    Used by visualize.py / evaluation, NOT by training.
    """
    prediction = prediction.view(s, s, B, 5)
    boxes = []
    for row in range(s):
        for col in range(s):
            for b in range(B):
                cx, cy, w, h, conf = prediction[row, col, b].tolist()
                if conf < conf_thresh:
                    continue
                boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf))
    # Highest confidence first
    boxes.sort(key=lambda box: box[4], reverse=True)
    return boxes

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