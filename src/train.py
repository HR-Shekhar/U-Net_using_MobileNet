# train.py
import os, time
from glob import glob
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from config import *
from model import get_model
from dataset import ImageMaskDataset, collect_pairs
from losses import combined_ce_dice
from utils import save_ckpt, iou_per_class
import numpy as np
from tqdm import tqdm

def prepare_dataloaders(data_root, batch_size=BATCH_SIZE, val_split=0.1):
    imgs, masks = collect_pairs(os.path.join(data_root, "images"), os.path.join(data_root, "masks"))
    dataset = ImageMaskDataset(imgs, masks)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

def freeze_encoder(model):
    # best-effort: freeze common encoders in smp or torchvision; user may need to adapt if custom layers
    for name, p in model.named_parameters():
        if "classifier" not in name and "decoder" not in name and "head" not in name:
            p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def evaluate(model, val_loader, device):
    model.eval()
    all_ious = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)['out'] if isinstance(model(imgs), dict) else model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            gts = masks.cpu().numpy()
            for p, g in zip(preds, gts):
                all_ious.append(iou_per_class(p, g, NUM_CLASSES))
    # compute mean IoU per class ignoring NaNs
    per_class = np.nanmean(np.array(all_ious), axis=0)
    return per_class, np.nanmean(per_class)

def train():
    device = DEVICE
    model = get_model().to(device)
    # freeze encoder first (transfer learning)
    freeze_encoder(model)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scaler = GradScaler()
    train_loader, val_loader = prepare_dataloaders(DATA_ROOT, BATCH_SIZE)
    best_miou = 0.0

    # stage 1: train decoder only
    print("Stage 1: training decoder only")
    for epoch in range(5):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage1-Epoch{epoch}")
        for imgs, masks in pbar:
            imgs = imgs.to(device); masks = masks.to(device)
            with autocast():
                out = model(imgs)
                logits = out['out'] if isinstance(out, dict) else out
                loss = combined_ce_dice(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())
        per_class, miou = evaluate(model, val_loader, device)
        print(f"Val mIoU: {miou:.4f}, per class: {per_class}")
        save_ckpt(os.path.join(RUNS, f"ckpt_stage1_epoch{epoch}.pth"), model, optimizer, epoch)

    # stage 2: unfreeze all and fine-tune
    print("Stage 2: fine-tuning all layers")
    unfreeze_all(model)
    optimizer = Adam(model.parameters(), lr=LR * 0.2)
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch{epoch}")
        for imgs, masks in pbar:
            imgs = imgs.to(device); masks = masks.to(device)
            with autocast():
                out = model(imgs)
                logits = out['out'] if isinstance(out, dict) else out
                loss = combined_ce_dice(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=loss.item())
        per_class, miou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} Val mIoU: {miou:.4f}")
        os.makedirs(RUNS, exist_ok=True)
        save_ckpt(os.path.join(RUNS, f"ckpt_epoch{epoch}.pth"), model, optimizer, epoch)
        if miou > best_miou:
            best_miou = miou
            save_ckpt(os.path.join(RUNS, f"ckpt_best.pth"), model, optimizer, epoch)
    print("Training finished")

if __name__ == "__main__":
    train()
