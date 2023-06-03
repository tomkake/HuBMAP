import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Loss(nn.Module):
    def __init__(self):

        super(Loss, self).__init__()

        self.diceloss = smp.losses.DiceLoss(mode="binary")
        self.binloss = smp.losses.SoftBCEWithLogitsLoss(
            reduction="mean", smooth_factor=0.1
        )

    def forward(self, output, mask):
        output = torch.squeeze(output)
        mask = torch.squeeze(mask)
        dice = self.diceloss(output, mask)
        bce = self.binloss(output, mask)

        loss = dice * 0.7 + bce * 0.3

        return loss


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
