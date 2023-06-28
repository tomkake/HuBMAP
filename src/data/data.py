import os
import random
import torch
import cv2
from PIL import Image
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .mosaic import Mosaic
from .transforms import get_cutmix_compose, get_transform, rand_bbox

from transformers import MaskFormerImageProcessor

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

class HuBMapDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        target_name,
        row_mame,
        cutmix_args,
        mosaic_args,
        augment=None,
        valid=False,
        pseudo=False,
        dataset_repeat=1,
        load = "npy"
    ) -> None:
        self.df = df
        self.unique_id = list(df["id"].unique())
        self.img_dir = img_dir
        self.valid = valid
        self.target_name = target_name
        self.row_mame = row_mame
        self.augment = augment
        self.cutmix_transform = get_cutmix_compose(768)
        self.cutmix_args = cutmix_args
        self.cutmix = self.cutmix_args.pop("enable") if not self.valid else False
        self.pseudo = pseudo
        self.dataset_repeat = dataset_repeat
        self.mosaic_args = mosaic_args
        self.mosaic = self.mosaic_args.pop("enable") if not self.valid else False
        if self.mosaic:
            self.mosaic_transform = Mosaic(**mosaic_args)
        self.load = load
        self.processor = MaskFormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)

    def prepare_data(self, idx, erosion=False):
        if idx < int(len(self.unique_id) * self.dataset_repeat):
            idx = idx % len(self.unique_id)
        target_id = self.unique_id[idx]
        img_path = os.path.join(
            self.img_dir, "train_images_class3", target_id + ".png"
        )
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        mask_path = os.path.join(
        self.img_dir, "train_masks_class3", target_id + ".npy"
        )
        mask = np.load(mask_path)
        class_id_map = mask[:,:,0]

        return img, mask[:,:,1],class_id_map

    def __getitem__(self, idx):
        # base image mask
        img, mask,class_id_map = self.prepare_data(idx)
        # Process annotations
        inst2class = {}
        class_labels = np.unique(class_id_map)
        # class_labels = np.array(list(class_labels))
        for label in class_labels:
            instance_ids = np.unique(mask[class_id_map == label])
            inst2class.update({i: label for i in instance_ids})

        if not self.valid:
            if self.cutmix:
                if not self.mosaic:
                    cutmix_idx = random.randint(0, self.__len__() - 1)
                    if cutmix_idx < int(len(self.unique_id) * self.dataset_repeat):
                        cutmix_idx = cutmix_idx % len(self.unique_id)
                    cutmix_image, cutmix_mask, tmp_map = self.prepare_data(idx=cutmix_idx)
                    class_labels
                    cutmix = self.cutmix_transform(
                        image=cutmix_image, mask=cutmix_mask, index=cutmix_idx
                    )
                    cutmix_image, cutmix_mask = cutmix["image"], cutmix["mask"]
                    lam_cutmix = np.random.beta(
                        self.cutmix_args["alpha"], self.cutmix_args["alpha"]
                    )
                    bbx1, bby1, bbx2, bby2, lam_cutmix = rand_bbox(
                        cutmix_image.shape, lam_cutmix
                    )
                    mixed_image, mixed_mask = img.copy(), mask.copy()
                    mixed_image[bbx1:bbx2, bby1:bby2] = cutmix_image[bbx1:bbx2, bby1:bby2]
                    mixed_mask[bbx1:bbx2, bby1:bby2] = cutmix_mask[bbx1:bbx2, bby1:bby2]
                    if self.augment is not None:
                        aug = self.augment(image=mixed_image, mask=mixed_mask)
                        img = aug["image"]
                        mask = aug["mask"]
                else:
                    img_cache = []
                    mask_cache = []
                    for i in range(3):
                        cutmix_idx = random.randint(0, self.__len__() - 1)
                        if cutmix_idx < int(len(self.unique_id) * self.dataset_repeat):
                            cutmix_idx = cutmix_idx % len(self.unique_id)
                        cutmix_image, cutmix_mask,tmp_map = self.prepare_data(idx=cutmix_idx)
                        if len(class_labels) < len(tmp_map):
                            class_labels = np.unique(class_id_map)
                        cutmix = self.cutmix_transform(
                            image=cutmix_image, mask=cutmix_mask, index=cutmix_idx
                        )
                        cutmix_image, cutmix_mask = cutmix["image"], cutmix["mask"]
                        lam_cutmix = np.random.beta(
                            self.cutmix_args["alpha"], self.cutmix_args["alpha"]
                        )
                        bbx1, bby1, bbx2, bby2, lam_cutmix = rand_bbox(
                            cutmix_image.shape, lam_cutmix
                        )
                        mixed_image, mixed_mask = img.copy(), mask.copy()
                        mixed_image[bbx1:bbx2, bby1:bby2] = cutmix_image[
                            bbx1:bbx2, bby1:bby2
                        ]
                        mixed_mask[bbx1:bbx2, bby1:bby2] = cutmix_mask[bbx1:bbx2, bby1:bby2]
                        if self.augment is not None:
                            aug = self.augment(image=mixed_image, mask=mixed_mask)
                            mixed_image = aug["image"]
                            mixed_mask = aug["mask"]
                        img_cache.append(mixed_image.permute(1, 2, 0).numpy())
                        mask_cache.append(mixed_mask.numpy())
                    result = self.mosaic_transform(
                        image=img, mask=mask, image_cache=img_cache, mask_cache=mask_cache
                    )
                    img, mask = result["image"], result["mask"]
            else:
                aug = self.augment(image=img, mask=mask)
                img = aug["image"]
                mask = aug["mask"]
        else:
            aug = self.augment(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]
        img = img.transpose(2,0,1)

        if class_labels.shape[0] == 1 and class_labels[0] == 0:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([img], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
          inputs = self.processor([img], [mask], instance_id_to_semantic_id=inst2class, return_tensors="pt")
          inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        return inputs

    def __len__(self):
        return len(self.unique_id) * self.dataset_repeat


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
        cutmix_args,
        pseudo,
        dataset_repeat,
        mosaic_args,
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
        self.mean = [0.62405882, 0.40748174, 0.68217106]
        self.std = [0.15112928, 0.20741797, 0.13163469]
        self.cutmix_args = cutmix_args
        self.pseudo = pseudo
        self.dataset_repeat = dataset_repeat
        self.mosaic_args = mosaic_args

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
            mosaic_args=self.mosaic_args,
            target_name=self.target_name,
            row_mame=self.row_mame,
            cutmix_args=self.cutmix_args,
            pseudo=self.pseudo,
            dataset_repeat=self.dataset_repeat,
        )
        self.vaild_dataset = HuBMapDataset(
            df=self.valid_df,
            img_dir=self.img_dir,
            augment=get_transform(
                self.mean, self.std, self.height, self.width, valid=True
            ),
            valid=True,
            mosaic_args=self.mosaic_args,
            target_name=self.target_name,
            row_mame=self.row_mame,
            cutmix_args=self.cutmix_args,
            dataset_repeat=1,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.vaild_dataset, batch_size=self.batch_size, num_workers=self.num_workers,collate_fn=collate_fn
        )
