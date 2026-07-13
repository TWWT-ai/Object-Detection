"""Shared helpers: annotation loading (format-adaptive), box math, prediction decoding.

Why format-adaptive annotation loading?  The dataset has an `annotation` folder per
gesture, but its file format was not documented.  Instead of hard-coding one format
(and crashing on another) we inspect the file extension and handle the three common
cases: mask images (png/jpg/bmp), JSON (bbox or polygon points), and plain-text boxes.
"""

import json          # to parse .json annotations if that is what the dataset uses
import os            # path handling
import numpy as np   # all mask / box math is done in numpy (fast, no GPU needed)
import torch         # only decode_predictions works on torch tensors
from PIL import Image, ImageDraw  # PIL instead of OpenCV: lighter dependency, ships with Colab

# extensions we treat as "the annotation is a segmentation mask image"
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def digits_key(filename):
    """Extract the digits from a filename to pair rgb/annotation files.

    Why digits and not the full name?  `rgb_003.png` and `mask_003.png` share no stem,
    but they DO share the index `003` — digits are the most robust pairing key.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]          # drop dir + extension
    digits = ''.join(ch for ch in stem if ch.isdigit())             # keep only 0-9 characters
    return digits if digits else stem                               # fall back to the stem


def mask_to_bbox(mask):
    """Derive a tight [x1, y1, x2, y2] box from a binary mask.

    Why derive the box from the mask instead of storing boxes separately?
    A tight box computed from the mask is ALWAYS consistent with the mask,
    so the detection and segmentation targets can never disagree.
    """
    ys, xs = np.nonzero(mask)                                       # coordinates of foreground pixels
    if len(xs) == 0:                                                # empty mask (bad annotation)
        h, w = mask.shape                                           # fall back to the full image
        return [0, 0, w - 1, h - 1]                                 # so training never crashes
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]  # tight box


def _find_in_json(obj, keys):
    """Recursively search a JSON structure for the first value under any of `keys`.

    Why recursive?  Annotation JSONs are often nested (e.g. {"shapes":[{"points":...}]})
    and we do not know the exact schema, so we search the whole tree.
    """
    if isinstance(obj, dict):                                       # dict: check keys then recurse
        for k, v in obj.items():
            if k.lower() in keys:                                   # case-insensitive key match
                return v
        for v in obj.values():                                      # not found at this level -> descend
            r = _find_in_json(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):                                     # list: recurse into every element
        for v in obj:
            r = _find_in_json(v, keys)
            if r is not None:
                return r
    return None                                                     # nothing found in this branch


def load_annotation(path, img_hw):
    """Return (binary mask HxW uint8, bbox [x1,y1,x2,y2] in pixels) for ANY supported format."""
    h, w = img_hw                                                   # target size = the RGB image size
    ext = os.path.splitext(path)[1].lower()                         # decide the parser by extension

    if ext in IMG_EXTS:                                             # --- case 1: mask image ---
        mask = np.array(Image.open(path).convert('L'))              # force single channel grayscale
        if mask.shape != (h, w):                                    # mask size can differ from RGB
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))  # NEAREST keeps labels binary
        mask = (mask > 0).astype(np.uint8)                          # any non-zero pixel = hand
        return mask, mask_to_bbox(mask)                             # tight box straight from the mask

    if ext == '.json':                                              # --- case 2: JSON annotation ---
        with open(path) as f:
            data = json.load(f)                                     # parse the whole document
        pts = _find_in_json(data, {'points', 'polygon', 'segmentation', 'landmarks'})
        if pts is not None:                                         # polygon points found
            pts = np.array(pts, dtype=np.float32).reshape(-1, 2)    # normalise to (N,2)
            if pts.max() <= 1.5:                                    # values look normalised (0..1)
                pts = pts * np.array([w, h], dtype=np.float32)      # scale to pixels
            canvas = Image.new('L', (w, h), 0)                      # blank mask
            ImageDraw.Draw(canvas).polygon([tuple(p) for p in pts], fill=1)  # fill the polygon
            mask = np.array(canvas, dtype=np.uint8)                 # back to numpy
            return mask, mask_to_bbox(mask)
        box = _find_in_json(data, {'bbox', 'box', 'rect', 'bounding_box'})
        if box is not None:                                         # only a box found -> rectangle mask
            b = [float(v) for v in box]
            if max(b) <= 1.5:                                       # normalised coords -> pixels
                b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
            if b[2] <= b[0] or b[3] <= b[1] or (b[0] + b[2] <= w and b[1] + b[3] <= h and b[2] < w * 0.9):
                b = [b[0], b[1], b[0] + b[2], b[1] + b[3]]          # heuristic: treat as x,y,w,h
            bbox = [int(max(0, b[0])), int(max(0, b[1])), int(min(w - 1, b[2])), int(min(h - 1, b[3]))]
            mask = np.zeros((h, w), dtype=np.uint8)                 # rectangle mask is the best
            mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1      # segmentation proxy a box gives us
            return mask, bbox
        raise ValueError(f'Unrecognised JSON annotation schema: {path}')

    if ext in {'.txt', '.csv'}:                                     # --- case 3: plain text box ---
        nums = [float(t) for t in open(path).read().replace(',', ' ').split() if
                t.replace('.', '', 1).replace('-', '', 1).isdigit()]  # grab all numbers in the file
        if len(nums) >= 5:                                          # YOLO txt: class cx cy w h (normalised)
            _, cx, cy, bw, bh = nums[:5]
            bbox = [int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                    int((cx + bw / 2) * w), int((cy + bh / 2) * h)]
        elif len(nums) == 4:                                        # bare box: assume x1 y1 x2 y2
            bbox = [int(v) for v in nums]
        else:
            raise ValueError(f'Cannot parse text annotation: {path}')
        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3])]
        mask = np.zeros((h, w), dtype=np.uint8)                     # again a rectangle mask proxy
        mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1] = 1
        return mask, bbox

    raise ValueError(f'Unsupported annotation format: {path}')      # fail loudly on unknown formats


def iou_xyxy(a, b):
    """IoU of two [x1,y1,x2,y2] boxes — the standard metric for box overlap."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])                     # intersection top-left
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])                     # intersection bottom-right
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)               # clamp: no overlap -> 0
    inter = iw * ih                                                 # intersection area
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])          # area of box a
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])          # area of box b
    union = area_a + area_b - inter                                 # inclusion-exclusion
    return inter / union if union > 0 else 0.0                      # avoid division by zero


def decode_predictions(pred, S, B, C, img_size):
    """Turn one raw YOLO output (S,S,B*5+C) into (class_id, score, [x1,y1,x2,y2] pixels).

    Why return only ONE box?  Every image in this dataset contains exactly one gesture,
    so instead of full NMS over multi-object outputs we simply keep the single highest
    (confidence x class-probability) box — simpler AND more accurate for this data.
    """
    boxes = pred[..., :B * 5].reshape(S, S, B, 5)                   # split the B boxes per cell
    cls_prob = pred[..., B * 5:]                                    # per-cell class probabilities
    conf = boxes[..., 4]                                            # objectness confidence per box
    best_cls_p, best_cls = cls_prob.max(dim=-1)                     # best class per cell
    scores = conf * best_cls_p.unsqueeze(-1)                        # YOLO score = conf * class prob
    flat = scores.flatten().argmax()                                # single best box in the image
    i = flat // (S * B)                                             # recover the grid row
    j = (flat % (S * B)) // B                                       # recover the grid column
    b = flat % B                                                    # recover which of the B boxes
    bx = boxes[i, j, b]                                             # the winning box vector
    cx = (j + bx[0].item()) / S * img_size                          # cell-relative x -> pixel centre x
    cy = (i + bx[1].item()) / S * img_size                          # cell-relative y -> pixel centre y
    bw = bx[2].item() * img_size                                    # width is predicted image-relative
    bh = bx[3].item() * img_size                                    # height is predicted image-relative
    box = [max(0, cx - bw / 2), max(0, cy - bh / 2),                # centre format -> corner format,
           min(img_size - 1, cx + bw / 2), min(img_size - 1, cy + bh / 2)]  # clipped to the image
    return int(best_cls[i, j]), float(scores[i, j, b]), box         # class id, score, pixel box