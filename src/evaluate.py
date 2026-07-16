import torch as th
import argparse

from torch.utils.data import DataLoader
from dataloader import get_dataLoaders, HandGestureDataset
from model import YOLOv1
from utils import compute_intersection_over_union, extract_gt_box, extract_yolo_prediction

# Must match training
S = 7
B = 2
C = 20


#---------------------------------------------------
# Evaluation loop
#---------------------------------------------------

@th.no_grad()
def evaluate(model, loader, device, n_classes=10, iou_thresh=0.5):
    model.eval()

    confusion = th.zeros(n_classes, n_classes, dtype=th.long)   # rows = truth, cols = prediction
    sum_det_iou, det_hits = 0.0, 0
    cls_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        predictions = model(images)          # [N, S*S*(B*5+C)]
        n = images.size(0)
        total += n

        for i in range(n):
            pred_box, pred_cls = extract_yolo_prediction(predictions[i], s=S, B=B, C=C)
            gt_box = extract_gt_box(targets["det"][i])
            true_cls = targets["Label"][i].item()

            # ---- Detection ----
            iou = compute_intersection_over_union(pred_box, gt_box).item()
            sum_det_iou += iou
            if iou >= iou_thresh:
                det_hits += 1

            # ---- Classification (YOLO class scores at the responsible cell) ----
            confusion[true_cls, pred_cls.item()] += 1
            if pred_cls.item() == true_cls:
                cls_correct += 1

    per_class_correct = confusion.diag().float()
    per_class_total = confusion.sum(dim=1).float()

    metrics = {
        "cls_acc": cls_correct / total,
        "cls_acc_per_class": (per_class_correct / per_class_total.clamp(min=1)).tolist(),
        "confusion": confusion,
        "det_mean_iou": sum_det_iou / total,
        "det_acc": det_hits / total,
        "n_samples": total,
    }
    return metrics


def print_report(m, iou_thresh=0.5):
    print("=" * 52)
    print(f"Evaluation on {m['n_samples']} held-out TEST images (unseen person)")
    print("=" * 52)
    print(f"[Detection]      mean IoU       : {m['det_mean_iou']:.4f}")
    print(f"                 acc @ IoU>={iou_thresh}: {m['det_acc']:.4f}")
    print(f"[Classification] accuracy       : {m['cls_acc']:.4f}")
    for c, acc in enumerate(m["cls_acc_per_class"]):
        print(f"                 G{c + 1:02d} accuracy   : {acc:.4f}")
    print("\nConfusion matrix (rows = truth G01..G10, cols = prediction):")
    header = "     " + " ".join(f"P{c + 1:02d}" for c in range(m["confusion"].size(1)))
    print(header)
    for r in range(m["confusion"].size(0)):
        row = " ".join(f"{v:3d}" for v in m["confusion"][r].tolist())
        print(f"G{r + 1:02d}  {row}")


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv1 detection model (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--weights", type=str, default="weights/best.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    # For an EXTERNAL test set: use every person in --data-root, no splitting
    parser.add_argument("--all-data", action="store_true")
    # MUST match the seed used in training, otherwise the person split changes
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.all_data:
        test_ds = HandGestureDataset(args.data_root, person_ids=None, augment=False, use_depth=True)
        test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        print(f"External test set: {len(test_ds)} images, ALL persons used")
    else:
        _, _, test_loader = get_dataLoaders(
            args.data_root,
            batch_size=args.batch_size,
            n_val_persons=args.n_val_persons,
            seed=args.seed,
            num_workers=args.num_workers,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

    model = YOLOv1(in_channels=4, split_size=S, num_boxes=B, num_classes=C).to(device)
    checkpoint = th.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded {args.weights} (epoch {checkpoint.get('epoch', '?')})")

    metrics = evaluate(model, test_loader, device, iou_thresh=args.iou_thresh)
    print_report(metrics, iou_thresh=args.iou_thresh)


if __name__ == "__main__":
    main()
