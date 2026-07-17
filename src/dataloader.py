import torch as th
from torch.utils.data import DataLoader
import PIL.Image as Image
import pathlib
import re
import numpy as np
import cv2
from utils import encode_yolo_target


def get_dataLoaders(root, batch_size=16, n_val_persons=5, seed=0, num_workers=2,
                    val_frac=0.1, test_frac=0.1, val_person=None):
    root = pathlib.Path(root)

    # Two supported layouts, auto-detected per file:
    #  merged:   root/GXX_x/clip/annotation/person__frame.png (person = prefix)
    #  original: root/<person>/GXX_x/clip/annotation/frame.png (person = folder)
    persons = set()
    for p in root.rglob("annotation/*png"):
        if "__" in p.stem:
            persons.add(p.stem.split("__")[0])
        else:
            rel = p.relative_to(root).parts
            persons.add(rel[0] if len(rel) >= 5 else root.name)
    persons = sorted(persons)
    if not persons:
        raise RuntimeError(f"No annotation found under {root}")

    # Explicitly named validation person: no shuffling, no test split —
    # train on everyone else. Used for the "train on A, validate on B" setup.
    if val_person is not None:
        if val_person not in persons:
            raise ValueError(f"val_person '{val_person}' not found, have: {persons}")
        val_ids = {val_person}
        test_ids = set()
        train_ids = set(persons) - val_ids
        return _build_loaders(root, train_ids, val_ids, test_ids, batch_size, num_workers)

    # Selecting people to load
    rng = np.random.default_rng(seed)
    rng.shuffle(persons)

    # 3-way person-level split (train/val/test) by fraction.
    # n_val_persons is IGNORED (kept in the signature so old callers don't break).
    # Split is by PERSON -> no leakage between splits.
    n = len(persons)
    # test_frac=0 means "no internal test person" — used for the final all-in
    # model when an EXTERNAL test set does the final scoring
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac))
    # keep at least 1 person for train; never negative (1-person smoke tests)
    n_val = max(0, min(n_val, n - n_test - 1))
    test_ids = set(persons[:n_test])
    val_ids = set(persons[n_test:n_test + n_val])
    train_ids = set(persons[n_test + n_val:])

    # Smoke-test fallbacks: tiny datasets can leave a split empty
    if len(train_ids) == 0:
        print("WARNING: too few persons for a real split, "
              "using ALL persons for train/val/test (smoke test only)")
        train_ids = val_ids = test_ids = set(persons)
    if len(val_ids) == 0:
        print("WARNING: no persons left for val, reusing TRAIN persons "
              "(overfit/smoke test only — val metrics are NOT generalization)")
        val_ids = train_ids

    return _build_loaders(root, train_ids, val_ids, test_ids, batch_size, num_workers)


def _build_loaders(root, train_ids, val_ids, test_ids, batch_size, num_workers):
    print(f"train persons: {sorted(train_ids)}")
    print(f"val persons:   {sorted(val_ids)}")
    print(f"test persons:  {sorted(test_ids)}")

    # Datasets
    train_ds = HandGestureDataset(root, person_ids=train_ids, augment=True, use_flip=True, use_depth=True)
    val_ds = HandGestureDataset(root, person_ids=val_ids, augment=False, use_depth=True)
    # No internal test persons -> reuse val as a placeholder so the 3-loader
    # interface stays intact; final scoring happens externally
    test_ds = HandGestureDataset(root, person_ids=test_ids, augment=False, use_depth=True) if test_ids else val_ds

    # Loading Data
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


class HandGestureDataset():
    def __init__(self, root, person_ids=None, transform=None, augment=False, use_flip=False, use_depth=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.augment = augment
        self.use_flip = use_flip
        self.use_depth = use_depth
        self.samples = []
        self.skipped = 0
        self.IMAGE_SIZE = 448

        for mask_path in sorted(self.root.rglob("annotation/*png")):
            # Breaking down path and folder
            clip_directory = mask_path.parent.parent
            gesture_directory = clip_directory.parent

            # Person: filename prefix (merged layout) or top-level folder
            # (original layout), matching the detection in get_dataLoaders
            stem = mask_path.stem
            if "__" in stem:
                person_id = stem.split("__")[0]
            else:
                rel = mask_path.relative_to(self.root).parts
                person_id = rel[0] if len(rel) >= 5 else self.root.name

            # If the person is not in the list then pass
            if person_ids is not None and person_id not in person_ids:
                continue

            # Finding the index of the gesture
            label = int(gesture_directory.name[1:3]) - 1

            # One folder holds many people's frame_005, so frame number alone is
            # NOT unique -> match by the FULL stem (merge renamed rgb/depth/
            # annotation to identical stems)
            rgb_path = self._find_by_stem(clip_directory / "rgb", stem)
            depth_path = self._find_by_stem(clip_directory / "depth", stem)

            # Skip (and count) incomplete samples instead of crashing:
            # one classmate's missing file should not block the whole cohort
            if rgb_path is None or depth_path is None:
                self.skipped += 1
                continue

            # Adding into the samples
            self.samples.append({
                "RGB": rgb_path,
                "Depth": depth_path,
                "Mask": mask_path,
                "Label": label,
                "Person": person_id
            })

        if self.skipped:
            print(f"WARNING: skipped {self.skipped} annotation(s) with no matching rgb/depth file")

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
        # Read file at position index
        s = self.samples[index]

        # 1. RGB into numpy
        rgb = np.array(Image.open(s["RGB"]).convert("RGB"))

        # 2. Depth into numpy — classmates exported different formats
        if s["Depth"].suffix == ".npy":
            depth = np.load(s["Depth"]).astype(np.float32)
        else:
            depth = cv2.imread(str(s["Depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]      # some exports save depth as 3-channel

        # 3. Mask into binary mask
        mask = np.array(Image.open(s["Mask"]).convert("L"))  # "L" means grey scale (0 - 255)
        mask = (mask > 127).astype(np.float32)               # separating hand (> 127) and background

        # Resizing all three images, image size is (width, height)
        rgb = cv2.resize(rgb, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        # Augmentation preventing overfitting forced to train
        if self.augment:
            # Horizontal flip
            if self.use_flip and np.random.rand() < 0.5:
                rgb = rgb[:, ::-1].copy()
                depth = depth[:, ::-1].copy()
                mask = mask[:, ::-1].copy()

            # Adjusting brightness (wide range for cross-person lighting robustness)
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.5, 1.5)
                rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # Random shift + scale: hand position/size changes every epoch, so
            # "recognize the background" stops working as a shortcut.
            # rgb / depth / mask MUST get the same transform.
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.2)
                tx = np.random.uniform(-0.08, 0.08) * self.IMAGE_SIZE
                ty = np.random.uniform(-0.08, 0.08) * self.IMAGE_SIZE
                M = np.float32([[scale, 0, tx], [0, scale, ty]])
                size = (self.IMAGE_SIZE, self.IMAGE_SIZE)
                rgb_a = cv2.warpAffine(rgb, M, size, flags=cv2.INTER_LINEAR)
                depth_a = cv2.warpAffine(depth, M, size, flags=cv2.INTER_LINEAR)
                # NEAREST for the mask: 0/1 labels must not be blended into 0.5
                mask_a = cv2.warpAffine(mask, M, size, flags=cv2.INTER_NEAREST)
                if mask_a.sum() > 0:      # keep only if the hand stayed in frame
                    rgb, depth, mask = rgb_a, depth_a, mask_a

        # Measuring the boundary box AFTER augmentation, so the box follows
        # whatever geometric transform was applied above
        rows, cols = np.where(mask == 1)
        if len(rows) == 0:
            raise ValueError(f"The mask is empty: {s['Mask']}")

        # x_min, y_min, x_max, y_max
        boundary_box = np.array([cols.min(), rows.min(), cols.max(), rows.max()], dtype=np.float32)
        # Normalizing and simplifying steps for YOLO encoding later
        boundary_box = boundary_box / self.IMAGE_SIZE

        # Corner box -> S x S x 5 YOLO grid, the format the loss expects
        det_target = encode_yolo_target(boundary_box)

        # Normalizing RGB
        rgb = rgb.astype(np.float32) / 255.0

        # Imputation with median
        valid = depth > 0
        if valid.any():
            depth[~valid] = np.median(depth[valid])

        # Normalizing depth: the cohort exported 8-bit depth maps (0-255),
        # keep a fallback for true 16-bit millimetre depth just in case
        if depth.max() > 255:
            depth = np.clip(depth, 0, 1500) / 1500.0
        else:
            depth = depth / 255.0

        # Transforming it into 3D matrix to match RGB
        # Then having a matrix with info of [R, G, B, depth]
        if self.use_depth:
            image = np.concatenate([rgb, depth[..., None]], axis=-1)
        else:
            image = rgb

        # Packing into what pytorch would expect
        # permute() converts into [channel, height, width], what nn.Conv2d expects
        image = th.from_numpy(image).permute(2, 0, 1).float()
        mask = th.from_numpy(mask).unsqueeze(0).float()

        # Keys must match compute_loss in train.py exactly: "det", "Mask", "Label"
        targets = {"det": det_target,
                   "Mask": mask,
                   "Label": th.tensor(s["Label"], dtype=th.long)
                   }

        return image, targets
