import torch as th
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import json

from dataloader import get_dataLoaders
from dataloader import HandGestureDataset
from model import HandGestureNet
from utils import compute_intersection_over_union
# Grid size
S = 7 

#---------------------------------------------------
# Loss Function
#---------------------------------------------------

def yolo_detection_loss(prediction, target, B=2,
                        lambda_coord=5.0, lambda_noobj=0.5):
    N = prediction.size(0)

    # [N, 7, 7, B * 5] -> [N, 7, 7, B, 5]
    raw_prediction = prediction.view(N, S, S, B, 5)

    # Keep coordinates and confidence in valid normalised ranges.
    # If added, apply the same conversion before visualising predictions.
    prediction = th.cat([
        th.sigmoid(raw_prediction[..., 0:2]),  # x, y
        th.sigmoid(raw_prediction[..., 2:4]),  # w, h
        th.sigmoid(raw_prediction[..., 4:5]),  # confidence
    ], dim=-1)

    object_mask = target[..., 4] > 0.5
    no_object_mask = ~object_mask

    # 1. Empty cells: both candidate boxes should have confidence = 0.
    empty_confidence = prediction[no_object_mask][..., 4]
    loss_no_object = F.mse_loss(
        empty_confidence,
        th.zeros_like(empty_confidence),
        reduction="sum",
    )

    if not object_mask.any():
        return lambda_noobj * loss_no_object / N

    # One object cell per image in this dataset.
    object_prediction = prediction[object_mask]  # [M, B, 5]
    object_target = target[object_mask]          # [M, 5]

    # 2. Pick the one candidate box responsible for this hand.
    ious = compute_intersection_over_union(
        object_prediction[..., :4],
        object_target[:, None, :4],
    )
    best_iou, best_idx = ious.max(dim=1)

    responsible_box = object_prediction[
        th.arange(object_prediction.size(0), device=prediction.device),
        best_idx,
    ]

    # 3. The other candidate box in an object cell is also "no object".
    responsible_mask = F.one_hot(best_idx, num_classes=B).bool()
    non_responsible_confidence = object_prediction[..., 4][~responsible_mask]
    loss_no_object += F.mse_loss(
        non_responsible_confidence,
        th.zeros_like(non_responsible_confidence),
        reduction="sum",
    )

    # 4. Coordinate loss for the responsible box only.
    loss_xy = F.mse_loss(
        responsible_box[:, :2], object_target[:, :2], reduction="sum"
    )
    loss_wh = F.mse_loss(
        th.sqrt(responsible_box[:, 2:4].clamp(min=1e-6)),
        th.sqrt(object_target[:, 2:4].clamp(min=1e-6)),
        reduction="sum",
    )

    # 5. Classic YOLO confidence target = IoU of the responsible box.
    loss_confidence = F.mse_loss(
        responsible_box[:, 4],
        best_iou.detach(),
        reduction="sum",
    )

    return (
        lambda_coord * (loss_xy + loss_wh)
        + loss_confidence
        + lambda_noobj * loss_no_object
    ) / N


def compute_loss(outputs, targets, segmentation_criterion, classification_criterion, lambda_seg=1.0, lambda_cls=1.0, lambda_det=1.0):
    # Computing loss function for each Head defined in models
    detection_pred, segmentation_pred, classification_pred = outputs

    # Loss per Head
    loss_detection = yolo_detection_loss(detection_pred, targets["det"])
    loss_segmentation = segmentation_criterion(segmentation_pred, targets["Mask"].float())
    loss_classification = classification_criterion(classification_pred, targets["Label"])

    total = lambda_det * loss_detection + lambda_seg * loss_segmentation + lambda_cls * loss_classification
    losses = {"det": loss_detection.item(), 
              "seg": loss_segmentation.item(),
              "cls": loss_classification.item(), 
              "total": total.item()
              }
    return total, losses


#---------------------------------------------------
# Train and validation
#---------------------------------------------------

def train_one_epoch(model, loader, optimizer, segmentation_criterion,
                    classification_criterion, device, lambda_segmentation,
                    lambda_classification, lambda_detection=1.0):
    model.train()
    running = {"det": 0.0, 
              "seg": 0.0,
              "cls": 0.0, 
              "total": 0.0
              }
    
    for images, targets in loader:
        # Copying what is on CPU onto GPU (RAM to VRAM)
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # Computing loss for each image
        outputs = model(images)
        loss, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection)

        # Cleaning the previously loaded data first
        optimizer.zero_grad()
        # The chain rule part, this tells you on which way you should adjust the direction
        # dloss/dw = dloss/dy * dy/dz * dz/dw
        loss.backward()
        # Using Adam formula to update the and override the old data
        optimizer.step()

        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

    n = len(loader)
    return {k: v / n for k, v in running.items()}

@th.no_grad()
def validate(model, loader, segmentation_criterion,
            classification_criterion, device, lambda_segmentation,
            lambda_classification, lambda_detection=1.0):
    model.eval()
    running = {"det": 0.0, 
              "seg": 0.0,
              "cls": 0.0, 
              "total": 0.0
              }
    correct, total_samples = 0, 0

    for images, targets in loader:
        # Copying what is on CPU onto GPU (RAM to VRAM)
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(images)
        _, parts = compute_loss(outputs, targets, segmentation_criterion,
                                   classification_criterion, lambda_segmentation,
                                   lambda_classification, lambda_detection)
        
        # Adding the loss to the training model
        for k in running:
            running[k] += parts[k]

        # Classification accuracy 
        pred_label = outputs[2].argmax(dim=1)
        correct += (pred_label == targets["Label"]).sum().item()
        total_samples += images.size(0)

    n = len(loader)
    metrics = {k: v / n for k, v in running.items()}
    metrics["cls_acc"] = correct / total_samples
    return metrics


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    # Like creating an empty sheet with all the data we are tracking
    parser = argparse.ArgumentParser(description="Train HandGestureNet (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-det", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # Checkpoint rule: cls_acc must improve AND seg/det val losses must stay
    # within (1 + best-tol) of the best they have ever been
    parser.add_argument("--best-tol", type=float, default=0.15)
    parser.add_argument("--out-dir", type=str, default="weights/")
    parser.add_argument("--seed", type=int, default=42)
    # Pack all the top argument into one
    args = parser.parse_args()

    # Keeping the training sample set the same every training at the start
    th.manual_seed(args.seed)
    # Selecting the available GPU device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Splitting data (3-way by person; test_loader is NEVER touched here —
    # it exists so the split is carved out consistently, evaluate.py uses it)
    train_loader, validation_loader, test_loader = get_dataLoaders(
        args.data_root,
        batch_size=args.batch_size,
        n_val_persons=args.n_val_persons,
        seed=args.seed,
        num_workers=args.num_workers,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    
    #Creating model, optimizer, loss
    model = HandGestureNet(in_channels=4, n_classes=10, B=2).to(device)
    segmentation_criterion = nn.BCEWithLogitsLoss()
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Creating location to store the outputs
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_seg_loss = float("inf")
    best_det_loss = float("inf")
    history = {"train_total": [], "val_total": [], "cls_acc": []}
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det)
        val_m = validate(model, validation_loader, segmentation_criterion,
                                  classification_criterion, device, args.lambda_seg, args.lambda_cls, args.lambda_det)
        scheduler.step()

        # so the curve survives even if Colab disconnects mid-training
        history["train_total"].append(train_m["total"])
        history["val_total"].append(val_m["total"])
        history["cls_acc"].append(val_m["cls_acc"])
        with open(output_directory / "history.json", "w") as f:
            json.dump(history, f)

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train {train_m['total']:.4f} "
              f"(det {train_m['det']:.3f} | seg {train_m['seg']:.3f} | cls {train_m['cls']:.3f})  "
              f"val {val_m['total']:.4f}  cls_acc {val_m['cls_acc']:.3f}")

        # Multi-task checkpoint rule (lexicographic with tolerance):
        #   primary   — val cls_acc must beat the best so far
        #   guardrail — val seg/det losses may fluctuate, but not more than
        #               (1 + best_tol) of the best they have EVER been
        # This stops us saving an epoch where classification improved but
        # segmentation/detection quietly collapsed.
        best_seg_loss = min(best_seg_loss, val_m["seg"])
        best_det_loss = min(best_det_loss, val_m["det"])
        seg_ok = val_m["seg"] <= best_seg_loss * (1 + args.best_tol) + 0.01
        det_ok = val_m["det"] <= best_det_loss * (1 + args.best_tol) + 0.05

        if val_m["cls_acc"] > best_val_acc and seg_ok and det_ok:
            best_val_acc = val_m["cls_acc"]
            th.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_m["total"],
                "val_cls_acc": best_val_acc,
                "val_seg_loss": val_m["seg"],
                "val_det_loss": val_m["det"],
                "args": vars(args),
            }, output_directory / "best.pth")
            print(f"  ↳ new best (cls_acc {best_val_acc:.3f}, "
                  f"seg {val_m['seg']:.3f}, det {val_m['det']:.3f}), checkpoint saved")
        elif val_m["cls_acc"] > best_val_acc:
            # cls improved but a guardrail failed — say so, don't save silently
            print(f"  ↳ cls_acc improved to {val_m['cls_acc']:.3f} but "
                  f"{'seg' if not seg_ok else 'det'} regressed beyond tolerance, NOT saved")

    # Save the best model/value overall
    th.save({"epoch": args.epochs, "model_state": model.state_dict()},
               output_directory / "last.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()