import os

import cv2
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .transforms import get_transform


class HuBMapDataset(Dataset):
    def __init__(
        self, df, img_dir, target_name, row_mame, augment=None, valid=False
    ) -> None:
        self.df = df
        self.img_dir = img_dir
        self.valid = valid
        self.target_name = target_name
        self.row_mame = row_mame
        self.augment = augment

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["id"] + ".tif")
        img = cv2.imread(img_path)

        if self.augment is not None:
            if (self.augment) is not None:
                img = self.augment(image=img)["image"]

        coord = self.df.iloc[idx]["coordinates"][0]
        # create mask array
        mask = np.zeros((512, 512), dtype=np.float32)
        points = np.array(coord)
        points = points.reshape((1, -1, 2))
        mask = cv2.fillPoly(mask, pts=points, color=(255))
        return img, mask

    def __len__(self):
        return len(self.df)


class HuBMapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        img_dir,
        height,
        width,
        target_name,
        row_mame,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.img_dir = img_dir
        self.height = height
        self.width = width
        self.target_name = target_name
        self.row_mame = row_mame
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def setup(self, stage):
        self.create_dataset()

    def create_dataset(self):
        self.train_dataset = HuBMapDataset(
            df=self.train_df,
            img_dir=self.img_dir,
            augment=get_transform(
                self.mean, self.std, self.height, self.width, valid=False
            ),
            valid=False,
            target_name=self.target_name,
            row_mame=self.row_mame,
        )
        self.vaild_dataset = HuBMapDataset(
            df=self.valid_df,
            img_dir=self.img_dir,
            augment=get_transform(
                self.mean, self.std, self.height, self.width, valid=True
            ),
            valid=True,
            target_name=self.target_name,
            row_mame=self.row_mame,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
