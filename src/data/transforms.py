import albumentations as albu
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2


def get_transform(
    mean=[0.68585333, 0.41604461, 0.63005896],
    std=[0.1331018, 0.21574782, 0.15331571],
    height=None,
    width=None,
    valid=False,
):
    ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
    ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
    if valid:
        return albu.Compose(
            [
                albu.Resize(height, width, always_apply=True, p=1),
                albu.Normalize(mean=ADE_MEAN, std=ADE_STD),
                # ToTensorV2(transpose_mask=True),
            ],
        )
    train_transform = [
        # albu.OneOf(
        #     [
        #         albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
        #         albu.RandomGamma(gamma_limit=(50, 150)),
        #         albu.NoOp(),
        #     ],
        #     p=0.5,
        # ),
        # base aug
        # albu.RandomRotate90(p=0.5),
        albu.augmentations.geometric.rotate.Rotate(limit = 270,p=5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        # distortion
        albu.OneOf(
            [
                albu.OpticalDistortion(p=0.5),
                albu.GridDistortion(p=0.5),
                albu.ElasticTransform(p=0.5),
                albu.NoOp(),
            ],
            p=0.5,
        ),
        # albu.CoarseDropout(p=0.5),
        # required aug
        albu.Resize(height, width, always_apply=True, p=1),
        albu.Normalize(mean=ADE_MEAN, std=ADE_STD),
        # ToTensorV2(transpose_mask=True),
    ]
    return albu.Compose(train_transform)


def get_cutmix_compose(crop_size):
    return albu.Compose(
        [
            albu.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                # pad_height_divisor=32,
                # pad_width_divisor=32,
                border_mode=4,
                value=None,
                mask_value=None,
                always_apply=True,
            ),
            # Sample Non-Empty mask with prob 0.5
            # Otherwise empty OR Non-Empty mask will be sampled
            albu.OneOrOther(
                first=albu.CropNonEmptyMaskIfExists(512, 512),
                second=albu.RandomCrop(512, 512),
                p=0.5,
            ),
        ]
    )


def rand_bbox(size, lam):
    """
    Retuns the coordinate of a random rectangle in the image for cutmix.
    Args:
        size (numpy ndarray [W x H x C]: Input size.
        lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.
    Returns:
        int: 4 coordinates of the rectangle.
        int: Proportion of the unmasked image.
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam
