import ast

import lightning as pl
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from data.data import HuBMapDataModule
from model.model import Model


def main():
    num_workers = 10
    batch_size = 32
    height = 512
    width = 512
    img_dir = "./data/train/"
    target_name = "coordinates"
    row_name = "id"
    gray_scale = False
    pretrained = True
    backbone = "resnet34"
    in_channels = 3
    epochs = 30
    df = pd.read_csv("./data/labels.csv")
    df["coordinates"] = [ast.literal_eval(d) for d in df["coordinates"]]
    model_checkpoint_dir = "outputs/model/"
    mode = "min"
    train_df, valid_df = train_test_split(df, test_size=0.33, random_state=42)
    model_checkpoint = ModelCheckpoint(
        dirpath=model_checkpoint_dir, mode=mode, save_weights_only=True
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=epochs,
        precision="32-true",
        callbacks=[model_checkpoint],
    )
    model = Model(
        backbone=backbone,
        gray_scale=gray_scale,
        pretrained=pretrained,
        in_channels=in_channels,
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
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
    pass
