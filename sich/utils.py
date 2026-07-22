import os
import random

import torch


def seed_everything(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_dice(
    logits: torch.Tensor, masks: torch.Tensor, num_classes: int, ignore_bg: bool = True
) -> float:
    """Calculates the macro-averaged multiclass Dice coefficient."""
    preds = torch.argmax(logits, dim=1)
    dice_scores = []

    start_class = 1 if ignore_bg else 0
    for c in range(start_class, num_classes):
        pred_c = preds == c
        mask_c = masks == c

        intersection = (pred_c & mask_c).sum().float()
        union = pred_c.sum().float() + mask_c.sum().float()

        if union > 0:
            dice_scores.append((2.0 * intersection) / union)

    return torch.stack(dice_scores).mean().item() if dice_scores else 0.0
