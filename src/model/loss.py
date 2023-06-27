import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):

        super(Loss, self).__init__()

        self.diceloss = smp.losses.DiceLoss(mode="binary")
        self.jloss = smp.losses.JaccardLoss(mode="binary")
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(
            reduction="mean", smooth_factor=0.1
        )
        self.focalloss = smp.losses.FocalLoss(mode="binary", reduction="mean")

    def forward(self, output, mask):
        dice = self.diceloss(output, mask)
        bce = self.binloss(output, mask)
        # jloss = self.jloss(output, mask)

        total_loss = dice * 0.6 + bce * 0.4

        return {"loss": total_loss, "dice_loss": dice, "bce": bce}


class DiceCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.0):

        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        y_pred = torch.round((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))

        intersection = (y_true * y_pred).sum()

        dice = (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

        return dice
