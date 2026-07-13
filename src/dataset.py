"""Dataset scanning, loading and YOLOv1 target encoding.

Design decisions (and why):
* We WALK the extracted zip instead of hard-coding paths, because zips often extract
  with an extra top-level folder — walking finds the class folders wherever they are.
* Boxes are derived from the segmentation masks (see utils.mask_to_bbox), so the two
  tasks can never have inconsistent labels.
* The `depth` folder is deliberately NOT used in this baseline: the pretrained ResNet
  backbone expects 3-channel RGB, and with only ~60 images the ImageNet-pretrained
  features help far more than a 4th input channel trained from scratch would.
* The train/val split is done PER CLASS, because a plain random 80/20 split of ~60
  images can easily leave a class with zero validation samples.
"""

import os                     # directory walking
import random                 # augmentation coin flips + split shuffling
import numpy as np            # image/mask arrays
import torch                  # tensors
from PIL import Image         # image IO (lighter than OpenCV, preinstalled on Colab)
from torch.utils.data import Dataset  # base class for PyTorch datasets
from torchvision import transforms    # colour jitter + normalisation

from . import config                                  # all hyper-parameters
from .utils import IMG_EXTS, digits_key, load_annotation  # shared helpers

# ImageNet statistics — required because the ResNet18 backbone was pretrained with them
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def find_class_dirs(root):
    """Find every folder that contains both an `rgb` and an `annotation` subfolder."""
    class_dirs = []                                                  # collected class folders
    for dirpath, dirnames, _ in os.walk(root):                       # walk the whole tree
        lower = {d.lower() for d in dirnames}                        # case-insensitive check
        if any('rgb' in d for d in lower) and any('annot' in d for d in lower):
            class_dirs.append(dirpath)                               # this folder IS a class folder
    return sorted(class_dirs)                                        # sorted -> stable class indices


def scan_dataset(root):
    """Build (samples, class_names): each sample pairs one rgb image with its annotation."""
    class_dirs = find_class_dirs(root)                               # e.g. .../G01_call ... G10_three
    assert class_dirs, f'No class folders with rgb+annotation found under {root}'
    class_names = []                                                 # human-readable class names
    samples = []                                                     # list of dicts (one per image)
    for cls_idx, cdir in enumerate(class_dirs):                      # enumerate -> class index 0..C-1
        base = os.path.basename(cdir)                                # e.g. 'G01_call'
        name = base.split('_', 1)[1] if '_' in base else base        # 'call' (drop the G01_ prefix)
        class_names.append(name)
        subdirs = {d.lower(): os.path.join(cdir, d)                  # map lowercase name -> path
                   for d in os.listdir(cdir)
                   if os.path.isdir(os.path.join(cdir, d))}
        rgb_dir = next(p for n, p in subdirs.items() if 'rgb' in n)  # locate the rgb folder
        ann_dir = next(p for n, p in subdirs.items() if 'annot' in n)  # locate the annotation folder
        rgb_files = sorted(f for f in os.listdir(rgb_dir)            # only image files, sorted for
                           if os.path.splitext(f)[1].lower() in IMG_EXTS)  # deterministic order
        ann_files = sorted(os.listdir(ann_dir))                      # annotations, any extension
        ann_by_key = {digits_key(f): f for f in ann_files}           # pair by shared digits (e.g. 003)
        for pos, rf in enumerate(rgb_files):                         # one sample per rgb image
            af = ann_by_key.get(digits_key(rf))                      # try the digits key first
            if af is None and pos < len(ann_files):                  # fall back to positional pairing
                af = ann_files[pos]                                  # (both lists are sorted)
            assert af is not None, f'No annotation found for {rf}'
            samples.append({'rgb': os.path.join(rgb_dir, rf),        # store absolute paths + label
                            'ann': os.path.join(ann_dir, af),
                            'cls': cls_idx})
    return samples, class_names


def split_per_class(samples, val_fraction, seed):
    """Split samples into train/val PER CLASS so every class appears in both sets."""
    rng = random.Random(seed)                                        # local RNG -> reproducible split
    by_class = {}                                                    # group sample indices by class
    for idx, s in enumerate(samples):
        by_class.setdefault(s['cls'], []).append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():                               # for each gesture class...
        rng.shuffle(idxs)                                            # shuffle within the class
        n_val = max(1, int(round(len(idxs) * val_fraction)))         # at least 1 val sample per class
        val_idx += idxs[:n_val]                                      # first n_val -> validation
        train_idx += idxs[n_val:]                                    # the rest -> training
    return train_idx, val_idx


def encode_yolo_target(bbox, cls_idx):
    """Encode ONE ground-truth box into the YOLOv1 target tensor (S, S, 5+C).

    Layout per cell: [x_cell, y_cell, w, h, objectness, one-hot classes].
    x_cell/y_cell are the box centre RELATIVE TO ITS GRID CELL (this is what makes
    YOLO's coordinates learnable in [0,1]); w/h are relative to the WHOLE image.
    """
    t = torch.zeros(config.S, config.S, 5 + config.C)                # start from an all-empty grid
    x1, y1, x2, y2 = bbox                                            # pixel corner coordinates
    cx = (x1 + x2) / 2 / config.IMG_SIZE                             # normalised centre x in [0,1]
    cy = (y1 + y2) / 2 / config.IMG_SIZE                             # normalised centre y in [0,1]
    w = (x2 - x1) / config.IMG_SIZE                                  # normalised width
    h = (y2 - y1) / config.IMG_SIZE                                  # normalised height
    j = min(config.S - 1, int(cx * config.S))                        # grid column of the centre
    i = min(config.S - 1, int(cy * config.S))                        # grid row of the centre
    t[i, j, 0] = cx * config.S - j                                   # centre x relative to cell (0..1)
    t[i, j, 1] = cy * config.S - i                                   # centre y relative to cell (0..1)
    t[i, j, 2] = w                                                   # image-relative width
    t[i, j, 3] = h                                                   # image-relative height
    t[i, j, 4] = 1.0                                                 # this cell contains an object
    t[i, j, 5 + cls_idx] = 1.0                                       # one-hot class label
    return t


class GestureDataset(Dataset):
    """Returns (image tensor, yolo target, mask tensor) for one sample."""

    def __init__(self, samples, indices, augment):
        self.items = [samples[i] for i in indices]                   # keep only this split's samples
        self.augment = augment                                       # flip/jitter only for training
        # colour jitter changes pixels but NOT geometry, so mask/box stay valid — that is
        # why it is safe to apply it to the image alone (a rotation would not be)
        self.jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # match backbone pretraining

    def __len__(self):
        return len(self.items)                                       # dataset size for the DataLoader

    def __getitem__(self, idx):
        item = self.items[idx]                                       # the sample record
        img = Image.open(item['rgb']).convert('RGB')                 # force 3 channels (some pngs are RGBA)
        w0, h0 = img.size                                            # original size (needed to scale boxes)
        mask, bbox = load_annotation(item['ann'], (h0, w0))          # format-adaptive annotation load
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)  # smooth resize for photos
        mask = Image.fromarray(mask * 255).resize(                   # NEAREST for masks: bilinear would
            (config.IMG_SIZE, config.IMG_SIZE), Image.NEAREST)       # create invalid in-between labels
        sx = config.IMG_SIZE / w0                                    # horizontal scale factor
        sy = config.IMG_SIZE / h0                                    # vertical scale factor
        bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]  # scale the box the same way
        if self.augment and random.random() < 0.5:                   # 50% horizontal flip (hands are
            img = img.transpose(Image.FLIP_LEFT_RIGHT)               # left/right symmetric, so this is
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)             # a label-preserving augmentation)
            x1, x2 = bbox[0], bbox[2]                                # mirror the box x-coordinates
            bbox[0] = config.IMG_SIZE - 1 - x2
            bbox[2] = config.IMG_SIZE - 1 - x1
        if self.augment:
            img = self.jitter(img)                                   # photometric augmentation (image only)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0  # HWC uint8 -> CHW float
        img_t = self.normalize(img_t)                                # ImageNet normalisation
        mask_t = (torch.from_numpy(np.array(mask)) > 0).float().unsqueeze(0)  # (1,H,W) binary target
        target = encode_yolo_target(bbox, item['cls'])               # YOLO grid target
        return img_t, target, mask_t