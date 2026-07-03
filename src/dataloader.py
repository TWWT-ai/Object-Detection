import torch as th
from torch.utils.data import Dataset
import os
import PIL.Image as Image
import csv
from typing import Callable, Optional, Tuple, Union, List
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import pathlib
import numpy as np
import cv2

class HandGestureDataset():
    def __init__(self, root, person_ids=None, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []
        self.IMAGE_SIZE = 448

        for mask_path in sorted(self.root.rglob("annotation/*png")):
            # Breaking down path and folder
            clip_directory = mask_path.parent.parent
            gesture_directory = clip_directory.parent
            person_directory = gesture_directory.parent
            person_id = person_directory.name

            # If the person is not in the list then pass
            if person_id is not None and person_id not in person_ids:
                continue

            # Finding the index of the gesture 
            label = int(gesture_directory.name[1:3]) - 1

            # Building the path if in same folder / directory
            rgb_path = clip_directory / "rgb" / mask_path.name
            depth_path = clip_directory / "depth" / (mask_path.name + ".npy")

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
        pass

    def get_dataloader(self):
        pass

