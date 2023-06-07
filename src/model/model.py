import lightning as pl
import segmentation_models_pytorch as smp
import torch

from .loss import Loss


class Model(pl.LightningModule):
    """Model

    Args:
        backbone (str): choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        gray_scale (bool): use grayscale image in training model
        pretrained (bool): use `imagenet` pre-trained weights for encoder initialization
        classes (int): model output channels (number of classes in your dataset)
        in_channels (int): model input channels (1 for gray-scale images, 3 for RGB, etc.)
        lr (float): training learning rate
    """

    def __init__(
        self,
        backbone="resnet34",
        gray_scale=False,
        pretrained=True,
        classes=1,
        in_channels=3,
        lr=1e-3,
        activation="sigmoid",
    ) -> None:
        super(Model, self).__init__()
        self.backbone = backbone
        self.gray_scale = gray_scale
        self.pretrained = pretrained
        self.classes = classes
        self.in_channels = in_channels
        self.lr = lr
        self.activation = activation

        self.create_model()
        self.loss = Loss()

    def create_model(self):
        self.model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.classes,
            activation=self.activation,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = self.loss(out, mask)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss = self.loss(out, mask)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.decoder.parameters(), "lr": 5e-5},
                {"params": self.model.encoder.parameters(), "lr": 8e-5},
            ]
        )
        # TODO: add scheduler
        # https://lightning.ai/docs/pytorch/latest/notebooks/lightning_examples/barlow-twins.html?highlight=dataset
        # warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         linear_warmup_decay(warmup_steps),
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }
        return optimizer
