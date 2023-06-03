from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np


class HuBMapDataset(Dataset):
    def __init__(self, df, img_dir, valid=False) -> None:
        self.df = df
        self.img_dir = img_dir
        self.valid = valid

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]["id"] + ".tif")
        img = cv2.imread(img_path)

        mask = self.df.iloc[idx]["mask"]
        return img, mask

    def __len__(self):
        return len(self.df)
