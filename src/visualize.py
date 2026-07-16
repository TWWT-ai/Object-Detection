import torch as th
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # No display needed: always render to files (works on Colab / ssh)
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataloader import get_dataLoaders, HandGestureDataset
from model import YOLOv1
from utils import extract_gt_box, extract_yolo_prediction
from evaluate import evaluate

IMAGE_SIZE = 448
S = 7
B = 2
C = 20


#---------------------------------------------------
# Drawing helpers
#---------------------------------------------------

def center_box_to_rect(box, size=IMAGE_SIZE):
    """(cx, cy, w, h) normalized -> (x_min, y_min, w, h) in pixels for
    matplotlib's Rectangle patch."""
    cx, cy, w, h = [float(v) * size for v in box]
    return cx - w / 2, cy - h / 2, w, h


def draw_sample(image, det_target, prediction, true_label, out_path):
    """One panel: RGB + ground truth box (solid) + predicted box (dashed)."""
    rgb = image[:3].permute(1, 2, 0).cpu().numpy()
    pred_box, pred_cls = extract_yolo_prediction(prediction, s=S, B=B, C=C)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    x, y, w, h = center_box_to_rect(extract_gt_box(det_target))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime",
                               linewidth=2, label="ground truth"))
    x, y, w, h = center_box_to_rect(pred_box)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red",
                               linewidth=2, linestyle="--", label="prediction"))
    ax.legend(loc="upper right", fontsize=8)
    correct = pred_cls.item() == true_label
    ax.set_title(f"T: G{true_label + 1:02d}  P: G{pred_cls.item() + 1:02d}",
                 color=("green" if correct else "red"))
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_predictions(model, loader, device, out_dir, n_samples=8):
    """Run the model on the first n images and save one png each."""
    model.eval()
    saved = 0
    with th.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            predictions = model(images)

            for i in range(images.size(0)):
                out_path = out_dir / f"sample_{saved:02d}.png"
                draw_sample(images[i].cpu(), targets["det"][i].cpu(), predictions[i].cpu(),
                            targets["Label"][i].item(), out_path)
                saved += 1
                if saved >= n_samples:
                    print(f"Saved {saved} sample visualizations to {out_dir}")
                    return


def plot_confusion_matrix(confusion, out_path, n_classes=10):
    """Heatmap: darker cell = more samples. Diagonal should light up."""
    confusion = confusion.numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    labels = [f"G{c + 1:02d}" for c in range(n_classes)]
    ax.set_xticks(range(n_classes), labels, rotation=45)
    ax.set_yticks(range(n_classes), labels)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix")

    threshold = confusion.max() / 2 if confusion.max() > 0 else 1
    for r in range(n_classes):
        for c in range(n_classes):
            ax.text(c, r, str(confusion[r, c]), ha="center", va="center", fontsize=8,
                    color="white" if confusion[r, c] > threshold else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def plot_loss_curves(history_path, out_path):
    """Detection loss curves from the json that train.py dumps."""
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"No {history_path} found, skipping loss curves")
        return

    with open(history_path) as f:
        history = json.load(f)

    train_key = "train_det" if "train_det" in history else "train_total"
    val_key = "val_det" if "val_det" in history else "val_total"

    epochs = range(1, len(history[train_key]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, history[train_key], label="train detection")
    ax.plot(epochs, history[val_key], label="val detection")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("YOLOv1 detection loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curves to {out_path}")


#---------------------------------------------------
# Main method
#---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv1 results (CW1)")
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--weights", type=str, default="weights/best.pth")
    parser.add_argument("--history", type=str, default="weights/history.json")
    parser.add_argument("--out-dir", type=str, default="figures/")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-val-persons", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    # For an EXTERNAL test set: use every person in --data-root, no splitting
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Loaded {args.weights}")

    # 1. Per-image predictions: gt vs predicted box + class in the title
    show_predictions(model, test_loader, device, out_dir, n_samples=args.n_samples)

    # 2. Confusion matrix heatmap (reuses evaluate() so numbers always match)
    metrics = evaluate(model, test_loader, device)
    plot_confusion_matrix(metrics["confusion"], out_dir / "confusion_matrix.png")

    # 3. Detection loss curves
    plot_loss_curves(args.history, out_dir / "loss_curves.png")


if __name__ == "__main__":
    main()
