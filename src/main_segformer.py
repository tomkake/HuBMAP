import ast
import datetime
import lightning as pl
import pandas as pd
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split

import random
import numpy as np
import os
from data.data import HuBMapDataModule
from model.model import Segformer

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()
    architecuture = "UNET"
    num_workers = 10
    batch_size = 8
    height = 512
    width = 512
    img_dir = "./data/"
    target_name = "blood_vessel"
    row_name = "id"
    gray_scale = False
    pretrained = "noisy-student"
    backbone = "timm-efficientnet-b3"
    activation = None
    in_channels = 3
    epochs = 150
    dataset_repeat = 8
    warm_up_epohs = 15
    classes = 1
    threshold = 0.5
    model_mode = "binary"
    precision = "16-mixed"  # default 32-true
    accumulate_grad_batches = 4
    swa_lrs = 1e-4
    annealing_epochs = 5
    pseudo = False
    cutmix_args = {
        "prob": 0.50,
        "enable": True,
        "alpha": 1.0,
        "height": height,
        "width": width,
    }
    mosaic_args = {"enable": False, "height": height, "width": width, "p": 0.5}
    df = pd.read_csv("./data/labels_1class.csv")
    meta = pd.read_csv("./data/tile_meta.csv")
    unique_id = list(meta[meta["dataset"] == 1]["id"].unique())
    random.shuffle(unique_id)
    all_len = len(unique_id)
    replace_index = int(all_len * 0.8)
    train_id = unique_id[:replace_index]
    valid_id = unique_id[replace_index+1:]
    assert set(train_id) != set(valid_id)
    # pseudo_label_df = pd.read_csv("./data/dataset3_2stage.csv")
    # df = pd.merge(df,pseudo_label_df,how = "outer")
    today = datetime.datetime.now()
    dir_key = today.strftime("%Y%m%d%H%M%S")
    model_checkpoint_dir = f"outputs/{architecuture}/{backbone}/{dir_key}"
    mode = "min"
    # train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)
    # train_df = pd.merge(train_df, pseudo_label_df, how="outer").reset_index(drop=True)
    # valid_df = valid_df.reset_index(drop=True)
    train_df = df[df["id"].isin(train_id)].reset_index(
        drop=True
    )
    valid_df = df[df["id"].isin(valid_id)].reset_index(
        drop=True
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=model_checkpoint_dir,
        mode=mode,
        save_weights_only=True,
        monitor="val_loss",
        save_last=True,
    )
    wandb_logger = WandbLogger(project="HubMap", name=dir_key)
    swa_callback = StochasticWeightAveraging(
        swa_lrs=swa_lrs, annealing_epochs=annealing_epochs
    )
    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    # )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        precision=precision,
        callbacks=[model_checkpoint, lr_monitor],
        logger=wandb_logger,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    model = Segformer(
        backbone=backbone,
        gray_scale=gray_scale,
        pretrained=pretrained,
        in_channels=in_channels,
        warm_up_epochs=warm_up_epohs,
        epochs=epochs,
        classes=classes,
        mode=model_mode,
        activation=activation,
        threshold=threshold,
    )
    datamodule = HuBMapDataModule(
        train_df=train_df,
        valid_df=valid_df,
        img_dir=img_dir,
        height=height,
        width=width,
        target_name=target_name,
        row_mame=row_name,
        batch_size=batch_size,
        num_workers=num_workers,
        cutmix_args=cutmix_args,
        pseudo=pseudo,
        dataset_repeat=dataset_repeat,
        mosaic_args=mosaic_args,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
    pass
