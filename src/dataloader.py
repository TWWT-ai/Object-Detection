import torch as th
from torch.utils.data import DataLoader
import PIL.Image as Image
import pathlib
import re
import numpy as np
import cv2
from utils import encode_yolo_target


def get_dataLoaders(root, batch_size=16, n_val_persons=5, seed=0, num_workers=2,
                    val_frac=0.1, test_frac=0.1):
    root = pathlib.Path(root)

    # Persons come from the RGB files now (not annotation), so persons that only
    # have un-annotated frames still show up in the split.
    persons = sorted({p.stem.split("__")[0] for p in root.rglob("rgb/*.png")})
    if not persons:
        raise RuntimeError(f"No rgb frames found under {root}")

    # Selecting people to load
    rng = np.random.default_rng(seed)
    rng.shuffle(persons)

    # 3-way person-level split (train/val/test) by fraction.
    n = len(persons)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac))
    n_val = min(n_val, n - n_test - 1)             # keep at least 1 person for train
    test_ids = set(persons[:n_test])
    val_ids = set(persons[n_test:n_test + n_val])
    train_ids = set(persons[n_test + n_val:])

    if len(train_ids) == 0:
        print("WARNING: too few persons for a real split, "
              "using ALL persons for train/val/test (smoke test only)")
        train_ids = val_ids = test_ids = set(persons)

    print(f"train persons: {sorted(train_ids)}")
    print(f"val persons:   {sorted(val_ids)}")
    print(f"test persons:  {sorted(test_ids)}")

    # TRAIN: use ALL frames (annotated_only=False) -> classification sees every
    # rgb+depth image; detection/segmentation only train on the annotated ones.
    train_ds = HandGestureDataset(root, person_ids=train_ids, augment=True, use_flip=True,
                                  use_depth=True, annotated_only=False)
    # VAL / TEST: annotated_only=True so evaluate.py (which needs boxes + masks)
    # and the seg/det guardrail keep working on real ground truth.
    val_ds = HandGestureDataset(root, person_ids=val_ids, augment=False,
                                use_depth=True, annotated_only=True)
    test_ds = HandGestureDataset(root, person_ids=test_ids, augment=False,
                                 use_depth=True, annotated_only=True) if test_ids else val_ds

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


class HandGestureDataset():
    def __init__(self, root, person_ids=None, transform=None, augment=False, use_flip=False,
                 use_depth=False, annotated_only=True):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.augment = augment
        self.use_flip = use_flip
        self.use_depth = use_depth
        # annotated_only=True  -> classic behaviour: only frames that have a mask
        # annotated_only=False -> use EVERY rgb frame; mask may be None
        self.annotated_only = annotated_only
        self.samples = []
        self.skipped = 0
        self.n_annotated = 0
        self.n_unannotated = 0
        self.IMAGE_SIZE = 448

        # Enumerate over RGB frames (not annotations) so un-annotated frames are included
        for rgb_path in sorted(self.root.rglob("rgb/*.png")):
            clip_directory = rgb_path.parent.parent      # .../Gxx/clipYY
            gesture_directory = clip_directory.parent    # .../Gxx

            stem = rgb_path.stem                          # e.g. 25040826_Guo__frame_005
            person_id = stem.split("__")[0]               # 25040826_Guo

            # If the person is not in this split, skip
            if person_ids is not None and person_id not in person_ids:
                continue

            # Gesture label from the folder name, e.g. "G01_call" -> 0
            label = int(gesture_directory.name[1:3]) - 1

            # Depth is REQUIRED (the model always takes a depth channel)
            depth_path = self._find_by_stem(clip_directory / "depth", stem)
            if depth_path is None:
                self.skipped += 1
                continue

            # Mask is OPTIONAL now
            mask_path = self._find_by_stem(clip_directory / "annotation", stem)

            # If we only want annotated frames, skip the ones without a mask
            if self.annotated_only and mask_path is None:
                continue

            if mask_path is None:
                self.n_unannotated += 1
            else:
                self.n_annotated += 1

            self.samples.append({
                "RGB": rgb_path,
                "Depth": depth_path,
                "Mask": mask_path,          # may be None
                "Label": label,
                "Person": person_id
            })

        if self.skipped:
            print(f"WARNING: skipped {self.skipped} rgb frame(s) with no matching depth file")

        print(f"Dataset built: {len(self.samples)} samples "
              f"({self.n_annotated} annotated + {self.n_unannotated} unannotated)")

        if len(self.samples) == 0:
            raise RuntimeError(f"No File found in this root: {self.root}")

    @staticmethod
    def _find_by_stem(folder, stem):
        """Return the file in `folder` whose stem matches exactly, else None."""
        if not folder.is_dir():
            return None
        for p in sorted(folder.iterdir()):
            if p.stem == stem:
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        # 1. RGB into numpy
        rgb = np.array(Image.open(s["RGB"]).convert("RGB"))

        # 2. Depth into numpy
        if s["Depth"].suffix == ".npy":
            depth = np.load(s["Depth"]).astype(np.float32)
        else:
            depth = cv2.imread(str(s["Depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]

        # 3. Mask into binary mask -- OR an all-zero placeholder if there is none
        if s["Mask"] is not None:
            mask = np.array(Image.open(s["Mask"]).convert("L"))
            mask = (mask > 127).astype(np.float32)
            has_mask = 1.0
        else:
            mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
            has_mask = 0.0

        # Resizing all three
        rgb = cv2.resize(rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            # Horizontal flip
            if self.use_flip and np.random.rand() < 0.5:
                rgb = rgb[:, ::-1].copy()
                depth = depth[:, ::-1].copy()
                mask = mask[:, ::-1].copy()

            # Brightness
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.5, 1.5)
                rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # Random shift + scale
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.2)
                tx = np.random.uniform(-0.08, 0.08) * self.IMAGE_SIZE
                ty = np.random.uniform(-0.08, 0.08) * self.IMAGE_SIZE
                M = np.float32([[scale, 0, tx], [0, scale, ty]])
                size = (self.IMAGE_SIZE, self.IMAGE_SIZE)
                rgb_a = cv2.warpAffine(rgb, M, size, flags=cv2.INTER_LINEAR)
                depth_a = cv2.warpAffine(depth, M, size, flags=cv2.INTER_LINEAR)
                mask_a = cv2.warpAffine(mask, M, size, flags=cv2.INTER_NEAREST)
                # For annotated frames keep the transform only if the hand stayed in frame.
                # For un-annotated frames there is no hand-in-frame constraint.
                if has_mask == 0.0 or mask_a.sum() > 0:
                    rgb, depth, mask = rgb_a, depth_a, mask_a

        # Build the detection + segmentation targets.
        # Only annotated frames get a real box; un-annotated frames get a dummy
        # target that the loss will IGNORE (via has_mask).
        if has_mask == 1.0:
            rows, cols = np.where(mask == 1)
            if len(rows) == 0:
                # Mask got augmented out of frame -> treat this frame as unannotated
                has_mask = 0.0
                det_target = th.zeros(7, 7, 5)
            else:
                boundary_box = np.array([cols.min(), rows.min(), cols.max(), rows.max()],
                                        dtype=np.float32) / self.IMAGE_SIZE
                det_target = encode_yolo_target(boundary_box)
        else:
            det_target = th.zeros(7, 7, 5)

        # Normalizing RGB
        rgb = rgb.astype(np.float32) / 255.0

        # Depth imputation + normalization
        valid = depth > 0
        if valid.any():
            depth[~valid] = np.median(depth[valid])
        if depth.max() > 255:
            depth = np.clip(depth, 0, 1500) / 1500.0
        else:
            depth = depth / 255.0

        # Stack RGB + depth -> 4 channels
        if self.use_depth:
            image = np.concatenate([rgb, depth[..., None]], axis=-1)
        else:
            image = rgb

        image = th.from_numpy(image).permute(2, 0, 1).float()
        mask = th.from_numpy(mask).unsqueeze(0).float()

        targets = {"det": det_target,
                   "Mask": mask,
                   "Label": th.tensor(s["Label"], dtype=th.long),
                   # 1.0 = has a real box+mask, 0.0 = classification-only sample
                   "has_mask": th.tensor(has_mask, dtype=th.float32)
                   }

        return image, targets