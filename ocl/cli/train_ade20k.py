"""Train a semantic segmentation model on ADE20K using PyTorch Lightning."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torchvision.models.segmentation as tv_segmentation
from torchmetrics import JaccardIndex

from ocl.datasets import ADE20KDataModule


class ADE20KSegmentationModule(pl.LightningModule):
    """LightningModule wrapping a torchvision segmentation backbone."""

    def __init__(
        self,
        num_classes: int,
        *,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = tv_segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["out"]

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)
        metric = self.train_iou if stage == "train" else self.val_iou
        metric(preds, masks)

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=images.size(0),
        )
        self.log(
            f"{stage}_miou",
            metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=images.size(0),
        )
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeeplabV3 on ADE20K.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ADE20K_2021_17_01",
        help="Path to the ADE20K root directory.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=150,
        help="Only keep the first N ADE20K classes (background retained). Use -1 to keep all classes.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap on training samples for quick experiments.",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Optional cap on validation samples for quick experiments.",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--log-every-n-steps", type=int, default=25)
    parser.add_argument(
        "--default-root-dir",
        type=str,
        default=None,
        help="Optional trainer output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_classes: Optional[int]
    if args.max_classes is None or args.max_classes < 0:
        max_classes = None
    else:
        max_classes = args.max_classes

    datamodule = ADE20KDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        max_classes=max_classes,
        train_limit=args.limit_train,
        val_limit=args.limit_val,
        pin_memory=True,
    )
    datamodule.setup("fit")

    model = ADE20KSegmentationModule(
        num_classes=datamodule.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.default_root_dir,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
