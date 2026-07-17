import argparse
import json
from pathlib import Path

import torch as th
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import get_dataLoaders, HandGestureDataset
from loss import YOLOLoss
from model import YOLOv1


# These constants are fixed by the indexing used inside loss.py.
S = 7
B = 2
C = 20


def prepare_yolo_targets(detection_target, labels):
    """Convert the dataloader target into the 30-value layout used by YOLOLoss.

    The dataloader provides [cx, cy, w, h, objectness] at each of the 7 x 7
    cells and one gesture label per image. loss.py indexes a target as:
    [20 class scores, objectness, cx, cy, w, h, second-box fields].
    Therefore its hard-coded indices require C=20, even though this dataset has
    only ten gesture labels. Classes 0--9 hold the gesture one-hot vector and
    classes 10--19 stay zero.
    """
    batch_size = detection_target.size(0)
    target = detection_target.new_zeros(batch_size, S, S, B * 5 + C)

    # loss.py layout is [C classes, CONFIDENCE@20, x@21, y@22, w@23, h@24] —
    # confidence FIRST, box after. The dataloader stores [x, y, w, h, conf]
    # (box first), so the two slices must be reordered, not copied as-is.
    target[..., 20:21] = detection_target[..., 4:5]   # objectness -> index 20
    target[..., 21:25] = detection_target[..., 0:4]   # cx,cy,w,h  -> 21..24

    object_mask = detection_target[..., 4:5]
    class_target = F.one_hot(labels, num_classes=C).to(detection_target.dtype)
    target[..., :C] = object_mask * class_target[:, None, None, :]
    return target


def compute_loss(predictions, targets, detection_criterion):
    """Calculate the single detection loss supported by YOLOv1."""
    yolo_target = prepare_yolo_targets(targets["det"], targets["Label"])
    detection_loss = detection_criterion(predictions, yolo_target)
    return detection_loss, {"det": detection_loss.item(), "total": detection_loss.item()}


def train_one_epoch(model, loader, optimizer, detection_criterion, device):
    """Run one optimisation epoch for the single YOLO detection output."""
    model.train()
    running = {"det": 0.0, "total": 0.0}

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}

        predictions = model(images)
        loss, parts = compute_loss(predictions, targets, detection_criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in running:
            running[key] += parts[key]

    return {key: value / len(loader) for key, value in running.items()}


@th.no_grad()
def validate(model, loader, detection_criterion, device):
    """Evaluate the YOLO loss on the validation split without updating weights."""
    model.eval()
    running = {"det": 0.0, "total": 0.0}

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}

        predictions = model(images)
        _, parts = compute_loss(predictions, targets, detection_criterion)

        for key in running:
            running[key] += parts[key]

    return {key: value / len(loader) for key, value in running.items()}


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv1 detection model (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # Name a specific person for validation: train on everyone else, no test
    parser.add_argument("--val-person", type=str, default="")
    # Validate on a DIFFERENT dataset folder (all persons in it). Overrides
    # the val split carved from --data-root.
    parser.add_argument("--val-root", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="weights/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    th.manual_seed(args.seed)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, validation_loader, _ = get_dataLoaders(
        args.data_root,
        batch_size=args.batch_size,
        n_val_persons=args.n_val_persons,
        seed=args.seed,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        val_person=args.val_person or None,
    )

    # Cross-dataset validation: train on --data-root, validate on --val-root
    if args.val_root:
        val_ds = HandGestureDataset(args.val_root, person_ids=None, augment=False, use_depth=True)
        validation_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)
        print(f"External val root: {args.val_root} ({len(val_ds)} images, all persons)")

    # C=20 is required by the hard-coded indices in the supplied YOLOLoss.
    model = YOLOv1(
        in_channels=4,
        split_size=S,
        num_boxes=B,
        num_classes=C,
    ).to(device)
    detection_criterion = YOLOLoss(S=S, B=B, C=C)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_detection_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_total": [], "val_total": [], "train_det": [], "val_det": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, detection_criterion, device)
        validation_metrics = validate(model, validation_loader, detection_criterion, device)
        scheduler.step()

        history["train_total"].append(train_metrics["total"])
        history["val_total"].append(validation_metrics["total"])
        history["train_det"].append(train_metrics["det"])
        history["val_det"].append(validation_metrics["det"])
        with open(output_directory / "history.json", "w") as history_file:
            json.dump(history, history_file)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train detection {train_metrics['det']:.4f}  "
            f"validation detection {validation_metrics['det']:.4f}"
        )

        if validation_metrics["det"] < best_detection_loss:
            best_detection_loss = validation_metrics["det"]
            epochs_no_improve = 0
            th.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": validation_metrics["total"],
                    "val_det_loss": validation_metrics["det"],
                    "args": vars(args),
                },
                output_directory / "best.pth",
            )
            print(f"  -> new best detection loss, saved")
        else:
            epochs_no_improve += 1

        # if epochs_no_improve >= args.patience:
        #     print(f"Early stopping at epoch {epoch} (best detection loss {best_detection_loss:.4f})")
        #     break

    th.save(
        {"epoch": epoch, "model_state": model.state_dict(), "args": vars(args)},
        output_directory / "last.pth",
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
