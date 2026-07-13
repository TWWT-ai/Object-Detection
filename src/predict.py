"""Inference script: python -m src.predict --weights best_model.pt --image path.png

Loads the trained checkpoint, runs one image through the network and saves a
visualisation with the predicted class, bounding box and segmentation overlay.
"""

import argparse                          # CLI arguments
import numpy as np                       # array math for the overlay
import torch                             # model execution
from PIL import Image, ImageDraw         # drawing the result (no OpenCV dependency)
from torchvision import transforms       # same normalisation as training

from . import config                                     # hyper-parameters
from .dataset import IMAGENET_MEAN, IMAGENET_STD         # must match training exactly
from .model import YoloV1Gesture                         # network definition
from .utils import decode_predictions                    # grid output -> single box


@torch.no_grad()                          # inference needs no gradients
def predict(weights, image_path, out_path='prediction.png'):
    ckpt = torch.load(weights, map_location=config.DEVICE)   # load weights + class names
    model = YoloV1Gesture().to(config.DEVICE)                # rebuild the architecture
    model.load_state_dict(ckpt['model'])                     # restore trained parameters
    model.eval()                                             # inference mode (BN frozen)
    classes = ckpt['classes']                                # class names saved at train time

    img = Image.open(image_path).convert('RGB')              # load and force 3 channels
    img_r = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)  # network input size
    x = torch.from_numpy(np.array(img_r)).permute(2, 0, 1).float() / 255.0  # HWC -> CHW float
    x = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)  # identical preprocessing to training
    det, seg = model(x.unsqueeze(0).to(config.DEVICE))       # add batch dim and forward

    cls_id, score, box = decode_predictions(det[0].cpu(), config.S, config.B,
                                            config.C, config.IMG_SIZE)      # best box + class
    mask = (torch.sigmoid(seg[0, 0]).cpu().numpy() > 0.5)    # threshold mask probabilities at 0.5

    overlay = np.array(img_r).astype(np.float32)             # start from the resized photo
    overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5  # blend red where mask=1
    out = Image.fromarray(overlay.astype(np.uint8))          # back to PIL for drawing
    draw = ImageDraw.Draw(out)                               # drawing context
    draw.rectangle(box, outline=(0, 255, 0), width=3)        # green predicted box
    draw.text((box[0] + 4, max(0, box[1] - 14)),             # label just above the box
              f'{classes[cls_id]} {score:.2f}', fill=(0, 255, 0))
    out.save(out_path)                                       # save the visualisation
    print(f'class={classes[cls_id]} score={score:.3f} box={[round(v, 1) for v in box]}')
    print(f'saved {out_path}')
    return classes[cls_id], score, box, mask                 # also return raw results for reuse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=config.CHECKPOINT)  # trained checkpoint
    parser.add_argument('--image', required=True)                # image to run on
    parser.add_argument('--out', default='prediction.png')      # where to save the overlay
    args = parser.parse_args()
    predict(args.weights, args.image, args.out)