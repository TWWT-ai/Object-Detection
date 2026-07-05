import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dataloader import HandGestureDataset
from model import HandGestureNet

# Grid size
S = 7 

#---------------------------------------------------
# Loss Function
#---------------------------------------------------
def compute_intersection_over_union(box1, box2):
    # Computing left and right boundary for both boxes (cx, cy, w, h)
    b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2         # left bound
    b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2         # right bound
    b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    # Calculating intersection for height and width
    inter_w = (th.min(b1_x2, b2_x2) - th.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (th.min(b1_y2, b2_y2) - th.may(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    # Calculating union
    union = box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - inter
    return inter / (union + 1e-6)

def yolo_detection_loss(prediction, target, B=2, lambda_coord=5.0, lambda_noobj=0.5):
    """
        Steps:
        1. Split cells into two groups -> 
        2. Push confidence toward zero for no-object cells ->
        3. Responsible-box selection ->
        4. Computing the three loss term: xy, wh, conf ->
        5. Weighted sum
    """

    N = prediction.size(0)
    # Unzip the combined last index back to Box (B) and (x, y, w, h, confidence)
    prediction = prediction.view(N, S, S, B, 5)

    # Detect whether there is an object within the boxes
    object_mask = target[..., 4] > 0.5
    no_object_mask = ~object_mask             # Flipping the mask

    # Selecting total numbers of all the boxes with the object and the confidence of the object
    no_object_conf = prediction[no_object_mask][..., 4]
    # Creating a no_object_conf size like (assumed correct answer) but filled with 0, containing all cells without object
    # Anything not equal to 0 will be considered loss/wrong prediction, to reduce the falseness reflected on to the last model
    loss_no_object = F.mse_loss(no_object_conf, th.zeros_like(no_object_conf), reduction="sum")

    if object_mask.sum() == 0:
        # Safety check
        return lambda_coord * loss_no_object / N

    # If there is something within the box
    prediction_object = prediction[object_mask]
    target_object = target[object_mask]

    inter_over_unions = compute_intersection_over_union(prediction_object[..., 4], target_object[:, None, : 4])
    best_iou, best_idx = inter_over_unions.max(dim=1)

    # Fancy indexing: using the selected best 
    response = prediction_object[th.arrange(prediction_object.size(0)), best_idx]

    # Calculating loss 
    loss_xy = F.mse_loss(response[:, 0:2], target_object[:, 0:2], reduction="sum")
    # Square root would punish small boundary boxes more 
    loss_wh = F.mse_loss(
        th.sqrt(response[:, 2:4].clamp(min=1e-6)),
        th.sqrt(target_object[:, 2:4].clamp(min=1e-6)),
        reduction="sum",
    )
    loss_confidence = F.mse_loss(response[:, 4], best_iou.detach(), reduction="sum")

    # Returning Multi Part loss function
    return (lambda_coord * (loss_xy + loss_wh) + loss_confidence + lambda_noobj * loss_no_object) / N

def compute_loss(outputs, targets, segmentation_criterion, classification_criterion, lambda_seg=1.0, lambda_cls=1.0):
    # Computing loss function for each Head defined in models
    detection_pred, segmentation_pred, classification_pred = outputs
    
    # Loss per Head
    loss_detection = yolo_detection_loss(detection_pred, targets["det"])
    loss_segmentation = segmentation_criterion(segmentation_pred, targets["Mask"].float())
    loss_classification = classification_criterion(classification_pred, targets["Label"])

    total = loss_detection + lambda_seg + loss_segmentation + lambda_cls * loss_classification
    losses = {"det": loss_detection.item(), 
              "seg": loss_segmentation.item(),
              "cls": loss_classification.item(), 
              "total": total.item()
              }
    return total, losses


def train_one_epoch():
    pass

