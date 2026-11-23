import torch
import segmentation_models_pytorch as smp


class CombinedLoss(torch.nn.Module):
    def __init__(self, class_weights, weight_ce=1.0, weight_dice=1.0):
        super().__init__()

        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, preds, targets):
        loss_ce = self.ce(preds, targets)
        loss_dice = self.dice(preds, targets)

        return self.w_ce * loss_ce + self.w_dice * loss_dice
