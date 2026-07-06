import torch as th
from torch.utils.data import DataLoader
import PIL.Image as Image
import pathlib
import numpy as np
import cv2
from utils import encode_yolo_target


def get_dataLoaders(root, batch_size=16, n_val_persons=5, seed=0, num_workers=2):
    root = pathlib.Path(root)
    persons = sorted(p.name for p in root.iterdir() if p.is_dir())

    # Selecting people to load
    rng = np.random.default_rng(seed)
    rng.shuffle(persons)
    val_ids = set(persons[:n_val_persons])
    train_ids = set(persons[n_val_persons:])

    # Smoke-test fallback: too few persons to split -> everyone plays both roles
    if len(train_ids) == 0:
        print("WARNING: too few persons for a real split, "
              "using ALL persons for both train and val (smoke test only)")
        train_ids = val_ids = set(persons)

    print(f"val persons: {sorted(val_ids)}")

    # Datasets
    train_ds = HandGestureDataset(root, person_ids=train_ids, augment=True, use_depth=True)
    val_ds = HandGestureDataset(root, person_ids=val_ids, augment=False, use_depth=True)

    # Loading Data
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

class HandGestureDataset():
    def __init__(self, root, person_ids=None, transform=None, augment=False, use_flip=False, use_depth=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.augment = augment
        self.use_flip = use_flip
        self.use_depth = use_depth
        self.samples = []
        self.IMAGE_SIZE = 448

        for mask_path in sorted(self.root.rglob("annotation/*png")):
            # Breaking down path and folder
            clip_directory = mask_path.parent.parent
            gesture_directory = clip_directory.parent
            person_directory = gesture_directory.parent
            person_id = person_directory.name

            # If the person is not in the list then pass
            if person_ids is not None and person_id not in person_ids:
                continue

            # Finding the index of the gesture 
            label = int(gesture_directory.name[1:3]) - 1

            # Building the path if in same folder / directory
            rgb_path = clip_directory / "rgb" / mask_path.name
            depth_path = clip_directory / "depth" / mask_path.name

            # Safety check
            if not rgb_path.exists():
                raise FileExistsError(f"RGB does not exists in the folder: {rgb_path}")
            if not depth_path.exists():
                raise FileExistsError(f"Depth does not exists in the folder: {depth_path}")

            # Adding into the samples
            self.samples.append({
                "RGB": rgb_path,
                "Depth": depth_path,
                "Mask": mask_path,
                "Label": label,
                "Person": person_id
            })

        if len(self.samples) == 0:
            raise RuntimeError(f"No File found in this root: {self.root}")
            

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        # Read file at position index
        s = self.samples[index]

        # 1. RGB into numpy
        rgb = np.array(Image.open(s["RGB"]).convert("RGB"))

        # 2. Depth into numpy
        depth = cv2.imread(str(s["Depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # 3. Mask into binary mask
        mask = np.array(Image.open(s["Mask"]).convert("L"))  # "L" means grey scale (0 - 255)
        mask = (mask > 127).astype(np.float32)               # separating hand (> 127) and background

        label = s["Label"]
        
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

            # Adjusting brightness
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.7, 1.3)
                rgb = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Measuring the boundary box after resized
        rows, cols = np.where(mask==1)
        if len(rows) == 0:
            raise ValueError(f"The mask is empty: {s['Mask']}")
        
        # x_min, y_min, x_max, y_max
        boundary_box = np.array([cols.min(), rows.min(), cols.max(), rows.max()], dtype=np.float32)
        # Normalizing and simplifying steps for YOLO endocing later
        boundary_box = boundary_box / self.IMAGE_SIZE

        # Corner box -> S x S x 5 YOLO grid, the format the loss expects
        det_target = encode_yolo_target(boundary_box)

        # Normalizing RGB
        rgb = rgb.astype(np.float32) / 255.0

        # Imputation with median
        valid = depth > 0
        if valid.any():
            depth[~valid] = np.median(depth[valid])

        # Normalizing depth
        depth = np.clip(depth, 0, 1500) / 1500.0

        # Transforming it into 3D matrix to match RGB
        # Then having a matrix with info of [R, G, B, depth]
        if self.use_depth:
            image = np.concatenate([rgb, depth[..., None]], axis=-1)
        else:
            image = rgb

        # Packing into what pytorch would expect
        # permute() converts into [channel, height, width], what nn.Conv2d expects
        # unsqueeze() would insert 1 at index 0 to fit the size
        image = th.from_numpy(image).permute(2, 0, 1).float()
        mask = th.from_numpy(mask).unsqueeze(0).float()

        # Keys must match compute_loss in train.py exactly: "det", "Mask", "Label"
        # - "det" is the encoded YOLO grid computed above (this was missing!)
        # - image is already returned separately, no need to duplicate it here
        targets = {"det": det_target,
                "Mask": mask,
                "Label": th.tensor(s["Label"], dtype=th.long)
                }

        return image, targets

