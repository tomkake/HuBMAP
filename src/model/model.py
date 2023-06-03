import segmentation_models_pytorch as smp


class Model:
    """Model

    Args:
        backbone (str): choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        gray_scale (bool): use grayscale image in training model
        pretrained (bool): use `imagenet` pre-trained weights for encoder initialization
        classes (int): model output channels (number of classes in your dataset)
        in_channels (int): model input channels (1 for gray-scale images, 3 for RGB, etc.)
    """

    def __init__(
        self,
        backbone="resnet34",
        gray_scale=False,
        pretrained=True,
        classes=1,
        in_channels=3,
    ) -> None:
        self.backbone = backbone
        self.gray_scale = gray_scale
        self.pretrained = pretrained
        self.classes = classes
        self.in_channels = in_channels

    def create_model(self):
        model = smp.Unet(
            encoder_name=self.backbone,
            encoder_weights="imagenet" if self.pretrained else None,
            in_channels=self.in_channels,
            classes=self.classes,
        )
        return model
