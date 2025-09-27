# losses.py
import torch
import torch.nn.functional as F

def dice_loss(logits, target, eps=1e-6):
    # logits: (B,C,H,W) raw; target: (B,H,W) ints
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=probs.shape[1]).permute(0,3,1,2).float()
    inter = (probs * target_onehot).sum(dim=(2,3))
    union = probs.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    dice = 2.0 * inter / (union + eps)
    return 1.0 - dice.mean()

def combined_ce_dice(logits, target, ce_weight=1.0, dice_weight=1.0):
    ce = F.cross_entropy(logits, target)
    d = dice_loss(logits, target)
    return ce_weight * ce + dice_weight * d
