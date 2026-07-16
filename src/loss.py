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
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        iou_box1 = compute_intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_box2 = compute_intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = th.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = th.max(ious, dim=0)
        exist_box = target[..., 20].unsqueeze(3)      #identity of obj_i

        # ======================= Box coordinates ================================

        box_predictions = exist_box * ((
                                    best_box * predictions[..., 26:30] +
                                    (1 - best_box) * predictions[..., 21:25]
                                    ))
        box_targets = exist_box * target[..., 21:25]

        box_predictions[..., 2:4] = th.sign(box_predictions[..., 2:4]) * th.sqrt(th.abs(box_predictions[..., 2:4] + 1e-6))
        # N, S, S, 25
        box_targets[..., 2:4] = th.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(th.flatten(box_predictions, end_dim=-2),
                            th.flatten(box_predictions, end_dim=-2)
                            )

        # ======================= Object loss ================================
        pred_box = (best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21])
        object_loss = self.mse(
                            th.flatten(exist_box * pred_box),
                            th.flatten(exist_box * target[..., 20:21])
                            )

        # ======================= No Object loss ================================
        # Box 1
        no_object_loss = self.mse(
                            th.flatten((1 - exist_box) * predictions[..., 20:21], start_dim=1),
                            th.flatten((1 - exist_box)  * target[..., 20:21], start_dim=1)
                            )

        # Box 2
        no_object_loss += self.mse(
                            th.flatten((1 - exist_box) * predictions[..., 25:26], start_dim=1),
                            th.flatten((1 - exist_box)  * target[..., 20:21], start_dim=1)
                            )

        # ======================= Class Loss ================================

        class_loss = self.mse(
                            th.flatten(exist_box * predictions[..., :20], end_dim=2),
                            th.flatten(exist_box * target[..., :20], end_dim=2)
                            )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
