import os, random, time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Local modules
from dataset import find_pairs, HipMRI3DDataset, dice_per_class
from modules import ImprovedUNet3D, DiceLoss3D

# ----------------------------
# Repro / Device
# ----------------------------
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")

# ----------------------------
# Paths (edit if needed)
# ----------------------------
ROOT = "/home/groups/comp3710/HipMRI_Study_open"
IMG_DIR = os.path.join(ROOT, "semantic_MRs")
LAB_DIR = os.path.join(ROOT, "semantic_labels_only")

# ----------------------------
# Hyperparams (tune if needed)
# ----------------------------
PATCH_SIZE   = (256, 256, 128)  # crop/pad target volume
BATCH_SIZE   = 1
ACCUM_STEPS  = 4
BASE_CH      = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT      = 0.1
NUM_WORKERS  = 8
AMP          = torch.cuda.is_available()
SAVE_PATH    = "unet3d_hipmri_best.pt"

# ----------------------------
# Evaluation helper
# ----------------------------
def evaluate(model, loader, n_classes):
    model.eval()
    tot = np.zeros(n_classes, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for imgs, labs, _ in loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, 1)  # (B,D,H,W)

            pred_oh = F.one_hot(preds, num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
            tgt_oh  = F.one_hot(labs,  num_classes=n_classes).permute(0, 4, 1, 2, 3).float()

            d = dice_per_class(pred_oh, tgt_oh)
            tot += np.nan_to_num(d, nan=0.0)  # accumulate valid classes only
            count += 1

    # Average per class (skip NaNs)
    mean_per_class = np.nan_to_num(tot / max(1, count), nan=0.0).astype(np.float32)
    mean_dice = float(np.nanmean(mean_per_class))  # overall mean Dice (unused but handy)
    return mean_per_class


def main():
    # ================================================================
    # Build splits & DataLoaders
    # ================================================================
    pairs = find_pairs(IMG_DIR, LAB_DIR)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    train_set = HipMRI3DDataset(train_pairs, patch_size=PATCH_SIZE)
    val_set   = HipMRI3DDataset(val_pairs,   patch_size=PATCH_SIZE)
    test_set  = HipMRI3DDataset(test_pairs,  patch_size=PATCH_SIZE)

    NUM_CLASSES = train_set.num_classes

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1,         shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=1,         shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ================================================================
    # Train & Validate
    # ================================================================
    model = ImprovedUNet3D(in_ch=1, n_classes=NUM_CLASSES, base=BASE_CH, dropout=DROPOUT).to(device)
    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = DiceLoss3D(n_classes=NUM_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device='cuda') if AMP else None

    best_val = float('inf')

    print("Training ImprovedUNet3D...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Starting Epoch {epoch}/{NUM_EPOCHS}")

        model.train()
        running = 0.0
        steps = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        # ----------------------------
        # Training Loop
        # ----------------------------
        for i, (imgs, labs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [Training]")):
            imgs = imgs.to(device)
            labs = labs.to(device)

            def forward_pass():
                logits = model(imgs)
                loss = 0.5 * ce_loss(logits, labs) + 0.5 * dice_loss(logits, labs)
                return loss

            if AMP:
                with autocast(device_type='cuda'):
                    loss = forward_pass()
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                loss = forward_pass()
                (loss / ACCUM_STEPS).backward()

            running += loss.item()
            steps += 1

            # Gradient accumulation
            if steps % ACCUM_STEPS == 0:
                if AMP:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labs, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
                imgs = imgs.to(device)
                labs = labs.to(device)
                logits = model(imgs)
                loss = 0.5 * ce_loss(logits, labs) + 0.5 * dice_loss(logits, labs)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        dsc_val = evaluate(model, val_loader, NUM_CLASSES)

        t1 = time.time()
        print(f"\nEpoch {epoch:03d}/{NUM_EPOCHS} Completed  |  Duration: {(t1 - t0)/60:.2f} min")
        print(f"Train Loss: {running / max(1, steps):.4f}  |  Val Loss: {val_loss:.4f}")
        print("  Val DSC:   [", end="")
        print(", ".join(str(int(v)) if v == 0 else f"{v:.3f}" for v in np.round(dsc_val, 3)), end="]\n")

        # ----------------------------
        # Save best model
        # ----------------------------
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'num_classes': NUM_CLASSES,
                'base_ch': BASE_CH,
                'patch_size': PATCH_SIZE
            }, SAVE_PATH)
            print(f"Saved new best model to {SAVE_PATH} (Val Loss: {val_loss:.4f})\n")
        else:
            print("No improvement this epoch.\n")

    # ================================================================
    # Test Evaluation
    # ================================================================
    print("\n[TEST] Loading best checkpoint and evaluating...")
    ckpt = torch.load(SAVE_PATH, map_location=device)
    model = ImprovedUNet3D(in_ch=1, n_classes=ckpt['num_classes'], base=ckpt['base_ch'], dropout=DROPOUT).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    test_dsc = evaluate(model, test_loader, ckpt['num_classes'])
    print("Test DSC per class:", np.round(test_dsc, 4))
    all_ok = bool((test_dsc >= 0.70).all())
    print("All labels ≥ 0.70:", all_ok)

if __name__ == "__main__":
    main()