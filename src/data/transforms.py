import albumentations as albu
from albumentations import OneOf
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    height=None,
    width=None,
    valid=False,
):
    if valid:
        return albu.Compose(
            [
                albu.Resize(height, width, always_apply=True, p=1),
                albu.Normalize(mean=mean, std=std, p=1),
                ToTensorV2(),
            ]
        )
    train_transform = [
        # OneOf(
        #     [
        #         albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
        #         albu.RandomGamma(gamma_limit=(50, 150)),
        #         albu.NoOp(),
        #     ],
        #     p=0.5,
        # ),
        # albu.VerticalFlip(p=0.5),
        # albu.HorizontalFlip(p=0.5),
        # albu.Cutout(p=0.5),
        albu.Resize(height, width, always_apply=True, p=1),
        albu.Normalize(mean=mean, std=std, p=1),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)
