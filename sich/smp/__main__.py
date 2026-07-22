import argparse
import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sich.dataset import SegmentationDataset
from sich.transform import TestTransform, TrainTransform
from sich.utils import calculate_dice, seed_everything

# Configure concise, structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


class DeepLabV3Trainer:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir)

        self.logger.info("Initializing DeepLabV3Trainer on device: %s", self.device)

        # 1. Initialize Model
        self.model = smp.DeepLabV3Plus(
            encoder_name=args.encoder,
            encoder_weights=args.encoder_weights,
            in_channels=args.in_channels,
            classes=args.num_classes,
        ).to(self.device)

        # 2. Initialize DataLoaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        # 3. Initialize Optimizer & Scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # 4. Initialize Loss & AMP Scaler
        self.criterion = smp.losses.TverskyLoss(
            mode="multiclass",
            classes=range(1, args.num_classes),
            from_logits=True,
            alpha=0.3,
            beta=0.7,
        )
        self.scaler = GradScaler(enabled=args.use_amp)

        self.logger.info(
            "Model initialized with %d parameters",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Builds training and validation DataLoaders."""
        train_dataset = SegmentationDataset(
            root=self.args.train_data,
            split="train",
            transform=TrainTransform(mean=self.args.mean, std=self.args.std),
        )
        val_dataset = SegmentationDataset(
            root=self.args.val_data,
            split="val",
            transform=TestTransform(mean=self.args.mean, std=self.args.std),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        self.logger.info(
            "DataLoaders built | Train samples: %d | Val samples: %d",
            len(train_dataset),
            len(val_dataset),
        )

        return train_loader, val_loader

    def train_epoch(self, epoch: int) -> None:
        """Trains the model for a single epoch."""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}", unit="batch")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.args.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss, running_dice = 0.0, 0.0

        pbar = tqdm(self.val_loader, desc=f"Validation {epoch}/{self.args.epochs}", unit="batch")
        for images, masks in pbar:
            images, masks = images.to(self.device), masks.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.args.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            dice = calculate_dice(logits, masks, self.args.num_classes, ignore_bg=False)

            running_loss += loss.item()
            running_dice += dice

            pbar.set_postfix(
                {"val_loss": running_loss / (pbar.n + 1), "val_dice": running_dice / (pbar.n + 1)}
            )

        avg_loss = running_loss / len(self.val_loader)
        avg_dice = running_dice / len(self.val_loader)
        return avg_loss, avg_dice

    def save_checkpoint(self, epoch: int, val_loss: float, val_dice: float, is_best: bool) -> None:
        """Saves model state, optimizer state, and scaler state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "val_dice": val_dice,
            "args": vars(self.args),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.output_dir / "last_model.pth"
        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best model to {best_path}")

    def fit(self) -> None:
        seed_everything(self.args.seed)

        self.logger.info("Starting training for %d epochs", self.args.epochs)
        best_val_dice = float("inf")

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            self.scheduler.step()

            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice

            self.save_checkpoint(epoch, val_loss, val_dice, is_best)

            self.logger.info(
                "Epoch [%d/%d]: Train Loss: %.4f | Val Loss: %.4f | "
                "Val Dice: %.4f | Best Dice: %.4f",
                epoch,
                self.args.epochs,
                train_loss,
                val_loss,
                val_dice,
                best_val_dice,
            )

        self.logger.info("Training completed. Best validation dice: %.4f", best_val_dice)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ for semantic segmentation")

    # Dataset args
    parser.add_argument("--train-data", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation dataset")
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=[0.485, 0.456, 0.406],
        help="Mean for normalization",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[0.229, 0.224, 0.225],
        help="Standard deviation for normalization",
    )
    parser.add_argument("--num-classes", type=int, default=6, help="Number of segmentation classes")
    parser.add_argument("--in-channels", type=int, default=1, help="Number of input channels")

    # Model args
    parser.add_argument(
        "--encoder", type=str, default="resnet34", help="Encoder architecture for DeepLabV3+"
    )
    parser.add_argument(
        "--encoder-weights", type=str, default="imagenet", help="Pretrained weights for the encoder"
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training and validation"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for the optimizer"
    )
    parser.add_argument(
        "--use-amp", action="store_true", help="Use Automatic Mixed Precision for training"
    )

    # System args
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        trainer = DeepLabV3Trainer(args, LOGGER)
        trainer.fit()
    except Exception as e:
        LOGGER.exception("An error occurred during training: %s", e)
        raise
