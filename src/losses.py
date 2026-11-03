import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, eps=1e-8):
    """
    pred:  Bx1xHxW after sigmoid
    target: Bx1xHxW in {0,1}
    """
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean()


def bce_dice_loss(logits, targets):
    """
    logits: raw model output (no sigmoid)
    targets: binary masks (0 or 1)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    dice = dice_coefficient(probs, targets)

    return bce + (1 - dice)
