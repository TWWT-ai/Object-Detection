import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_intersection_over_union


class YOLOLoss(nn.Module):
    """YOLO-v1-style loss for a detection-only head with B boxes per grid cell.

    ``predictions`` is [N, S, S, B * 5] and ``target`` is [N, S, S, 5], where
    each target entry is [centre_x, centre_y, width, height, objectness].
    Classification is intentionally not included: HandGestureNet has a separate
    image-level classification head trained with CrossEntropyLoss.
    """

    def __init__(self, S=7, B=2, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        batch_size = predictions.size(0)
        expected_shape = (batch_size, self.S, self.S, self.B * 5)
        if tuple(predictions.shape) != expected_shape:
            raise ValueError(
                f"Expected detection predictions shaped {expected_shape}, "
                f"but received {tuple(predictions.shape)}."
            )
        if tuple(target.shape) != (batch_size, self.S, self.S, 5):
            raise ValueError(
                f"Expected detection targets shaped {(batch_size, self.S, self.S, 5)}, "
                f"but received {tuple(target.shape)}."
            )

        predicted_boxes = predictions.view(batch_size, self.S, self.S, self.B, 5)
        target_box = target[..., :4]
        object_mask = target[..., 4:5]

        # Choose the candidate box that currently overlaps the target best.
        ious = compute_intersection_over_union(
            predicted_boxes[..., :4], target_box.unsqueeze(-2)
        )
        best_iou, best_box_index = ious.max(dim=-1, keepdim=True)
        responsible_box = F.one_hot(best_box_index.squeeze(-1), num_classes=self.B).unsqueeze(-1)
        responsible_box = responsible_box.to(dtype=predictions.dtype)
        responsible_object = object_mask.unsqueeze(-2) * responsible_box

        responsible_prediction = (responsible_box * predicted_boxes[..., :4]).sum(dim=-2)
        xy_loss = F.mse_loss(
            object_mask * responsible_prediction[..., :2],
            object_mask * target_box[..., :2],
            reduction="sum",
        )
        wh_loss = F.mse_loss(
            object_mask * th.sign(responsible_prediction[..., 2:4])
            * th.sqrt(responsible_prediction[..., 2:4].abs() + 1e-6),
            object_mask * th.sqrt(target_box[..., 2:4].clamp_min(0.0)),
            reduction="sum",
        )

        # Only the responsible candidate predicts the IoU confidence in cells
        # containing an object. Every other candidate is trained toward zero.
        object_loss = F.mse_loss(
            responsible_object * predicted_boxes[..., 4:5],
            responsible_object * best_iou.detach().unsqueeze(-1),
            reduction="sum",
        )
        no_object_mask = 1.0 - responsible_object
        no_object_loss = F.mse_loss(
            no_object_mask * predicted_boxes[..., 4:5],
            th.zeros_like(predicted_boxes[..., 4:5]),
            reduction="sum",
        )

        return (
            self.lambda_coord * (xy_loss + wh_loss)
            + object_loss
            + self.lambda_noobj * no_object_loss
        ) / batch_size
