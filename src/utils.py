# utils.py
import numpy as np
import torch
import os

def iou_per_class(pred, target, num_classes):
    # pred, target are numpy arrays shape (H,W)
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious

def save_ckpt(path, model, optimizer=None, epoch=None):
    data = {"model_state": model.state_dict()}
    if optimizer is not None:
        data["opt_state"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch
    torch.save(data, path)

def load_ckpt(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "opt_state" in ckpt:
        optimizer.load_state_dict(ckpt["opt_state"])
    return ckpt
