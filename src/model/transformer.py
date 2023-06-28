import lightning as pl
import segmentation_models_pytorch as smp
import torch
from timm.scheduler import CosineLRScheduler
from transformers import MaskFormerForInstanceSegmentation
from .loss import Loss
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from transformers import MaskFormerImageProcessor

class MaskFormer(pl.LightningModule):
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
        id2label,
        backbone="resnet34",
        gray_scale=False,
        pretrained=True,
        classes=3,
        in_channels=3,
        lr=1e-3,
        activation="softmax",
        warm_up_epochs=5,
        threshold=None,
        epochs=30,
        mode="multiclass",
    ) -> None:
        super(MaskFormer, self).__init__()
        self.backbone = backbone
        self.gray_scale = gray_scale
        self.pretrained = pretrained
        self.classes = classes
        self.in_channels = in_channels
        self.lr = lr
        self.activation = activation
        self.warm_up_epochs = warm_up_epochs
        self.threshold = threshold
        self.epochs = epochs
        self.mode = mode
        self.metric = 1.0
        self.id2label = id2label
        self.processor = MaskFormerImageProcessor()

        self.create_model()
        self.loss = Loss()
        # loss
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # dice loss
        self.training_dice_step_outputs = []
        self.validation_dice_step_outputs = []
        # jcard loss
        # self.training_jcard_step_outputs = []
        # self.validation_jcard_step_outputs = []
        # binary cross entropy
        self.training_bce_step_outputs = []
        self.validation_bce_step_outputs = []
        # iou_score
        self.training_step_iou_score = []
        self.validation_step_iou_score = []
        # recall
        self.training_step_recall = []
        self.validation_step_recall = []
        # precision
        self.training_step_ap = []
        self.validation_step_ap = []

    def create_model(self):
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade",
                                                          id2label=self.id2label,
                                                          ignore_mismatched_sizes=True)

    def forward(self, pixel_values,mask_labels,class_labels):
        # x = x.permute(1, 0, 2, 3) # (C,N,H,W) -> # (N,C,H,W)
        return self.model(pixel_values = pixel_values,mask_labels = mask_labels,class_labels = class_labels)

    def training_step(self, batch, batch_idx):
        pixel_values=batch["pixel_values"]
        mask=[labels.cuda() for labels in batch["mask_labels"]]
        class_labels=[labels.cuda() for labels in batch["class_labels"]]
        out = self.forward(pixel_values,mask,class_labels)
        loss = self.loss(
            out, mask
        )  # {"loss": total_loss,"dice_loss":dice,"bce":bce,"jloss":jloss}
        self.training_step_outputs.append(loss["loss"])
        self.training_dice_step_outputs.append(loss["dice_loss"])
        self.training_bce_step_outputs.append(loss["bce"])
        # self.training_jcard_step_outputs.append(loss["jloss"])
        tp, fp, fn, tn = smp.metrics.get_stats(
            out,
            mask,
            mode=self.mode,
            threshold=self.threshold,
            num_classes=self.classes,
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.training_step_iou_score.append(iou_score)
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        self.training_step_recall.append(recall)
        idx = 0
        mtp = 0
        mfp = 0
        mfn = 0
        mtn = 0
        for threshold in range(50, 95, 5):
            threshold = threshold * 0.01
            tp, fp, fn, tn = smp.metrics.get_stats(
                out,
                mask,
                mode=self.mode,
                threshold=threshold,
                num_classes=self.classes,
            )
            idx += 1
            mtp += tp
            mfp += fp
            mfn += fn
            mtn += tn
        mtp = mtp / idx
        mfp = mfp / idx
        mfn = mfn / idx
        mtn = mtn / idx
        ap = smp.metrics.precision(mtp, mfp, mfn, mtn, reduction="micro-imagewise")
        self.training_step_ap.append(ap)
        return {
            "loss": loss["loss"],
            "iou_score": iou_score,
            "recall": recall,
            "average_precision": ap,
        }

    def validation_step(self, batch, batch_idx):
        b,c,h,w = batch["pixel_values"].shape
        pixel_values=batch["pixel_values"]
        mask=[labels.cuda() for labels in batch["mask_labels"]]
        class_labels=[labels.cuda() for labels in batch["class_labels"]]
        out = self.forward(pixel_values,mask,class_labels)
        results = self.processor.post_process_instance_segmentation(outputs=out, target_sizes=[(h,w)] * b)[0]
        # print(out.keys()) # odict_keys(['loss', 'class_queries_logits', 'masks_queries_logits', 'encoder_last_hidden_state', 'pixel_decoder_last_hidden_state', 'transformer_decoder_last_hidden_state'])
        print(out["loss"])
        print(out["class_queries_logits"])
        print(out["masks_queries_logits"])
        print(results["segmentation"].shape)
        loss = self.loss(
            out, mask
        )
        self.validation_step_outputs.append(loss["loss"])
        self.validation_dice_step_outputs.append(loss["dice_loss"])
        self.validation_bce_step_outputs.append(loss["bce"])
        # self.validation_jcard_step_outputs.append(loss["jloss"])
        tp, fp, fn, tn = smp.metrics.get_stats(
            out,
            mask,
            mode=self.mode,
            threshold=self.threshold,
            num_classes=self.classes,
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        self.validation_step_iou_score.append(iou_score)
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        self.validation_step_recall.append(recall)
        idx = 0
        mtp = 0
        mfp = 0
        mfn = 0
        mtn = 0
        for threshold in range(50, 95, 5):
            threshold = threshold * 0.01
            tp, fp, fn, tn = smp.metrics.get_stats(
                out,
                mask,
                mode=self.mode,
                threshold=threshold,
                num_classes=self.classes,
            )
            idx += 1
            mtp += tp
            mfp += fp
            mfn += fn
            mtn += tn
        mtp = mtp / idx
        mfp = mfp / idx
        mfn = mfn / idx
        mtn = mtn / idx
        ap = smp.metrics.precision(mtp, mfp, mfn, mtn, reduction="micro-imagewise")
        self.validation_step_ap.append(ap)
        return {
            "loss": loss["loss"],
            "iou_score": iou_score,
            "recall": recall,
            "average_precision": ap,
        }

    def on_train_epoch_end(self, mode="train"):
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.log(f"{mode}_loss", epoch_loss, on_epoch=True)
        self.training_step_outputs.clear()  # free memory
        epoch_dice_loss = torch.stack(self.training_dice_step_outputs).mean()
        self.log(f"{mode}_dice_loss", epoch_dice_loss, on_epoch=True)
        self.training_dice_step_outputs.clear()  # free memory
        epoch_bce_loss = torch.stack(self.training_bce_step_outputs).mean()
        self.log(f"{mode}_bce_loss", epoch_bce_loss, on_epoch=True)
        self.training_bce_step_outputs.clear()  # free memory
        # epoch_jcard_loss = torch.stack(self.training_jcard_step_outputs).mean()
        # self.log(f"{mode}_jcard_loss", epoch_jcard_loss, on_epoch=True)
        # self.training_jcard_step_outputs.clear()  # free memory
        epochs_iou = torch.stack(self.training_step_iou_score).mean()
        self.log(f"{mode}_iou", epochs_iou, on_epoch=True)
        self.training_step_iou_score.clear()  # free memory
        epochs_recall = torch.stack(self.training_step_recall).mean()
        self.log(f"{mode}_recall", epochs_recall, on_epoch=True)
        self.training_step_recall.clear()  # free memory
        epochs_ap = torch.stack(self.training_step_ap).mean()
        self.log(f"{mode}_average_precision", epochs_ap, on_epoch=True)
        self.training_step_ap.clear()  # free memory

    def on_validation_epoch_end(self, mode="val"):
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f"{mode}_loss", epoch_loss, on_epoch=True)
        self.validation_step_outputs.clear()  # free memory
        epoch_dice_loss = torch.stack(self.validation_dice_step_outputs).mean()
        self.log(f"{mode}_dice_loss", epoch_dice_loss, on_epoch=True)
        self.validation_dice_step_outputs.clear()  # free memory
        epoch_bce_loss = torch.stack(self.validation_bce_step_outputs).mean()
        self.log(f"{mode}_bce_loss", epoch_bce_loss, on_epoch=True)
        self.validation_bce_step_outputs.clear()  # free memory
        # epoch_jcard_loss = torch.stack(self.validation_jcard_step_outputs).mean()
        # self.log(f"{mode}_jcard_loss", epoch_jcard_loss, on_epoch=True)
        # self.validation_jcard_step_outputs.clear()  # free memory
        epochs_iou = torch.stack(self.validation_step_iou_score).mean()
        self.log(f"{mode}_iou", epochs_iou, on_epoch=True)
        self.validation_step_iou_score.clear()  # free memory
        epochs_recall = torch.stack(self.validation_step_recall).mean()
        self.log(f"{mode}_recall", epochs_recall, on_epoch=True)
        self.validation_step_recall.clear()  # free memory
        epochs_ap = torch.stack(self.validation_step_ap).mean()
        self.log(f"{mode}_average_precision", epochs_ap, on_epoch=True)
        self.validation_step_ap.clear()  # free memory

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        scheduler.step(self.current_epoch)

    def configure_optimizers(self):
        def func(step: int, max_steps=self.num_training_steps):
            return (1 - (step / max_steps)) ** 0.9

        optimizer = torch.optim.Adam(
            [
                # {"params": self.model.decoder.parameters(), "lr": 5e-7},
                {"params": self.model.parameters(), "lr": 1e-3},
            ]
        )
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=func)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
        #                                       max_lr=1e-3, epochs=self.epochs, steps_per_epoch=self.trainer.estimated_stepping_batches)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.epochs,
            lr_min=1e-7,
            warmup_t=self.warm_up_epochs,
            warmup_lr_init=5e-6,
            warmup_prefix=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs
    
    def get_mask(self,segmentation, segment_id):
        mask = (segmentation.cpu().numpy() == segment_id)
        visual_mask = (mask * 255).astype(np.uint8)
        visual_mask = Image.fromarray(visual_mask)
        return visual_mask