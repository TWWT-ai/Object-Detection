"""Central configuration for the YOLOv1 gesture detection + segmentation project.

Every hyper-parameter lives here so the dataset, model, loss and training code
all read from ONE place — changing a value here changes it everywhere.
"""

import torch  # needed only to auto-select the compute device below

# ---------------------------------------------------------------- YOLOv1 grid
S = 7            # YOLOv1 divides the image into an S x S grid (7 is the paper value;
                 # with one hand per image a coarse grid is enough and keeps the head small)
B = 2            # boxes predicted per grid cell (paper value; 2 lets the network
                 # specialise one box for wide and one for tall objects)
C = 10           # number of gesture classes (G01..G10 folders in the dataset)

# --------------------------------------------------------------------- images
IMG_SIZE = 448   # YOLOv1's native input resolution (448 = 7 * 64, so the ResNet
                 # /32 feature map is 14x14 and one extra stride-2 conv gives exactly 7x7)

# ------------------------------------------------------------------- training
BATCH_SIZE = 8       # small batch: the dataset is tiny (~60 images), and 8 fits any Colab GPU
EPOCHS = 60          # enough for the loss to plateau on a tiny dataset without wasting GPU time
LR = 1e-4            # AdamW learning rate; fine-tuning a pretrained backbone needs a small LR
WEIGHT_DECAY = 1e-4  # mild L2 regularisation — important because the dataset is tiny
VAL_FRACTION = 0.2   # 20% of each class is held out for validation (per-class, see dataset.py)
SEED = 42            # fixed seed so train/val split and results are reproducible

# --------------------------------------------------------------- loss weights
LAMBDA_COORD = 5.0   # paper value: boost box-coordinate loss (localisation matters most)
LAMBDA_NOOBJ = 0.5   # paper value: damp confidence loss of empty cells (48 of 49 cells
                     # are empty here, so without damping they would dominate the loss)
LAMBDA_SEG = 2.0     # weight of the segmentation loss relative to the detection loss
                     # (2.0 balances the two tasks — seg loss is per-pixel and already averaged)

# --------------------------------------------------------------------- device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # use the Colab GPU when present

# ----------------------------------------------------------------------- data
DATA_ROOT = 'data'            # folder where the dataset zip was extracted
CHECKPOINT = 'best_model.pt'  # where the best (highest val box-IoU) weights are saved