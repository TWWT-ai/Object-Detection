"""Training script: python -m src.train [--data data] [--epochs 60]

Trains the multi-task YOLOv1 model and saves the checkpoint with the best
validation box IoU (box IoU is chosen as the model-selection metric because
localisation is the hardest of the three tasks on this tiny dataset —
classification saturates early and would not discriminate between checkpoints).
"""

import argparse                      # command-line arguments
import random                        # seeding
import numpy as np                   # seeding + metric math
import torch                         # everything ML
from torch.utils.data import DataLoader  # batching

from . import config                                              # hyper-parameters
from .dataset import GestureDataset, scan_dataset, split_per_class  # data pipeline
from .loss import YoloV1Loss, seg_loss                            # the two losses
from .model import YoloV1Gesture                                  # the network
from .utils import decode_predictions, iou_xyxy                   # evaluation helpers


def set_seed(seed):
    """Seed every RNG so runs are reproducible (essential when comparing changes)."""
    random.seed(seed)                 # python RNG (used by the augmentations + split)
    np.random.seed(seed)              # numpy RNG
    torch.manual_seed(seed)           # torch CPU RNG
    torch.cuda.manual_seed_all(seed)  # torch GPU RNG


@torch.no_grad()                      # evaluation never needs gradients -> saves memory/time
def evaluate(model, loader, det_criterion):
    """Return (val loss, class accuracy, mean box IoU, mean mask IoU) on a loader."""
    model.eval()                                                   # switch off dropout/BN updates
    tot_loss, n_img = 0.0, 0                                       # running sums
    correct, box_ious, mask_ious = 0, [], []
    for img, target, mask in loader:                               # iterate validation batches
        img = img.to(config.DEVICE)                                # move to GPU
        target = target.to(config.DEVICE)
        mask = mask.to(config.DEVICE)
        det, seg = model(img)                                      # forward pass
        loss = det_criterion(det, target) + config.LAMBDA_SEG * seg_loss(seg, mask)
        tot_loss += loss.item() * img.shape[0]                     # de-average for correct mean later
        prob = torch.sigmoid(seg)                                  # mask probabilities
        for k in range(img.shape[0]):                              # per-image metrics
            n_img += 1
            cls_pred, _, box_pred = decode_predictions(            # best predicted box + class
                det[k].cpu(), config.S, config.B, config.C, config.IMG_SIZE)
            obj = (target[k, ..., 4] > 0.5).nonzero(as_tuple=False)[0]  # the single GT cell (i,j)
            i, j = int(obj[0]), int(obj[1])
            tcell = target[k, i, j]                                # GT vector in that cell
            cls_gt = int(tcell[5:].argmax())                       # GT class from the one-hot part
            cx = (j + float(tcell[0])) / config.S * config.IMG_SIZE  # decode GT box back to pixels
            cy = (i + float(tcell[1])) / config.S * config.IMG_SIZE  # (same maths as the encoder,
            bw = float(tcell[2]) * config.IMG_SIZE                   # inverted)
            bh = float(tcell[3]) * config.IMG_SIZE
            box_gt = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            correct += int(cls_pred == cls_gt)                     # classification accuracy count
            box_ious.append(iou_xyxy(box_pred, box_gt))            # localisation quality
            pm = (prob[k, 0] > 0.5).float()                        # binarise the predicted mask
            gm = mask[k, 0]                                        # GT mask
            inter = (pm * gm).sum().item()                         # mask IoU numerator
            union = pm.sum().item() + gm.sum().item() - inter      # mask IoU denominator
            mask_ious.append(inter / union if union > 0 else 1.0)  # both empty -> perfect match
    return (tot_loss / max(n_img, 1), correct / max(n_img, 1),
            float(np.mean(box_ious)), float(np.mean(mask_ious)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=config.DATA_ROOT)        # dataset root folder
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    args = parser.parse_args()

    set_seed(config.SEED)                                          # reproducibility first
    samples, class_names = scan_dataset(args.data)                 # find all rgb/annotation pairs
    print(f'Found {len(samples)} samples, classes: {class_names}')
    train_idx, val_idx = split_per_class(samples, config.VAL_FRACTION, config.SEED)
    train_ds = GestureDataset(samples, train_idx, augment=True)    # training set WITH augmentation
    val_ds = GestureDataset(samples, val_idx, augment=False)       # val set WITHOUT (deterministic eval)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)      # shuffle: decorrelate batches
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = YoloV1Gesture().to(config.DEVICE)                      # build + move the network
    det_criterion = YoloV1Loss()                                   # YOLOv1 detection loss
    # AdamW instead of the paper's SGD + hand-tuned LR schedule: AdamW adapts per-parameter
    # step sizes, which converges much faster on small datasets and needs no warmup tuning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    # cosine annealing smoothly decays the LR to ~0 — a simple, schedule that avoids
    # picking manual step milestones for an unknown dataset.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = 0.0                                                 # best val box IoU so far
    for epoch in range(1, args.epochs + 1):
        model.train()                                              # enable BN updates / augment path
        running = 0.0
        for img, target, mask in train_loader:                     # one pass over the training set
            img = img.to(config.DEVICE)
            target = target.to(config.DEVICE)
            mask = mask.to(config.DEVICE)
            det, seg = model(img)                                  # forward
            loss = det_criterion(det, target) + config.LAMBDA_SEG * seg_loss(seg, mask)
            optimizer.zero_grad()                                  # clear old gradients
            loss.backward()                                        # backprop through both heads
            optimizer.step()                                       # update weights
            running += loss.item() * img.shape[0]
        scheduler.step()                                           # decay the learning rate
        val_loss, acc, biou, miou = evaluate(model, val_loader, det_criterion)
        print(f'epoch {epoch:3d} | train {running / len(train_ds):.3f} | '
              f'val {val_loss:.3f} | acc {acc:.2f} | box IoU {biou:.3f} | mask IoU {miou:.3f}')
        if biou > best_iou:                                        # keep only the best checkpoint
            best_iou = biou
            torch.save({'model': model.state_dict(),               # weights
                        'classes': class_names},                   # class names travel with the model
                       config.CHECKPOINT)
            print(f'  saved new best (box IoU {biou:.3f})')
    print(f'Done. Best val box IoU: {best_iou:.3f} -> {config.CHECKPOINT}')


if __name__ == '__main__':
    main()