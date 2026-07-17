import torch as th
import torch.nn as nn
from utils import compute_intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=10):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(
            -1, self.S, self.S, self.B * 5 + self.C
        )

        # Layout:
        # [class scores (C), confidence_1, box_1 (x,y,w,h),
        #  confidence_2, box_2 (x,y,w,h)]
        confidence_1_index = self.C
        box_1_start = self.C + 1
        confidence_2_index = self.C + 5
        box_2_start = self.C + 6

        # Sigmoid bounds the predicted x,y offsets to (0,1) — the same range
        # as the cell-relative targets (YOLOv2's stability fix). w,h stay raw
        # (handled by the sign-sqrt trick below).
        box_1 = predictions[..., box_1_start:box_1_start + 4]
        box_1 = th.cat([th.sigmoid(box_1[..., :2]), box_1[..., 2:]], dim=-1)
        box_2 = predictions[..., box_2_start:box_2_start + 4]
        box_2 = th.cat([th.sigmoid(box_2[..., :2]), box_2[..., 2:]], dim=-1)
        target_box = target[..., box_1_start:box_1_start + 4]

        iou_box_1 = compute_intersection_over_union(box_1, target_box)
        iou_box_2 = compute_intersection_over_union(box_2, target_box)

        ious = th.stack([iou_box_1, iou_box_2], dim=0)
        iou_maxes, best_box = th.max(ious, dim=0)

        # The extra final dimension allows multiplication with x/y/w/h tensors.
        best_box = best_box.unsqueeze(3)
        exist_box = target[..., confidence_1_index:confidence_1_index + 1]

        # ---------------- Box-coordinate loss ----------------
        box_predictions = exist_box * (
            best_box * box_2 + (1 - best_box) * box_1
        )
        box_targets = exist_box * target_box

        # NOT in-place: slice-assignment would overwrite values autograd still
        # needs for backward (RuntimeError: modified by an inplace operation).
        # Build new tensors with cat instead.
        pred_wh = (
            th.sign(box_predictions[..., 2:4])
            * th.sqrt(th.abs(box_predictions[..., 2:4]) + 1e-6)
        )
        box_predictions = th.cat([box_predictions[..., 0:2], pred_wh], dim=-1)

        target_wh = th.sqrt(box_targets[..., 2:4].clamp(min=0.0))
        box_targets = th.cat([box_targets[..., 0:2], target_wh], dim=-1)

        box_loss = self.mse(
            th.flatten(box_predictions, end_dim=-2),
            th.flatten(box_targets, end_dim=-2),
        )

        # ---------------- Object-confidence loss ----------------
        predicted_confidence = (
            best_box * predictions[..., confidence_2_index:confidence_2_index + 1]
            + (1 - best_box)
            * predictions[..., confidence_1_index:confidence_1_index + 1]
        )

        object_loss = self.mse(
            th.flatten(exist_box * predicted_confidence),
            th.flatten(exist_box * iou_maxes.unsqueeze(3).detach()),
        )

        # ---------------- No-object-confidence loss ----------------
        no_object_loss = self.mse(
            th.flatten(
                (1 - exist_box)
                * predictions[..., confidence_1_index:confidence_1_index + 1],
                start_dim=1,
            ),
            th.flatten(
                (1 - exist_box)
                * target[..., confidence_1_index:confidence_1_index + 1],
                start_dim=1,
            ),
        )

        no_object_loss += self.mse(
            th.flatten(
                (1 - exist_box)
                * predictions[..., confidence_2_index:confidence_2_index + 1],
                start_dim=1,
            ),
            th.flatten(
                (1 - exist_box)
                * target[..., confidence_1_index:confidence_1_index + 1],
                start_dim=1,
            ),
        )

        # ---------------- Class loss ----------------
        class_loss = self.mse(
            th.flatten(exist_box * predictions[..., :self.C], end_dim=2),
            th.flatten(exist_box * target[..., :self.C], end_dim=2),
        )

        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
