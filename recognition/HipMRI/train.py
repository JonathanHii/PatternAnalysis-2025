import os, random, time, math
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
NUM_EPOCHS   = 15
LR           = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT      = 0.1
NUM_WORKERS  = 2
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