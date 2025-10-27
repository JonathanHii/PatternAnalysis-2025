# ------------------------------------------
# Minimal viewer for a 3D Improved U-Net .pt
# Set the variables in the CONFIG block and run.

import os
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from modules import ImprovedUNet3D

# ============== CONFIG (edit these) ==============
CKPT_PATH  = "unet3d_hipmri_best.pt"
IMAGE_PATH = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B006_Week0_LFOV.nii.gz"
LABEL_PATH = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz"  # or None

Z = 30   # axial slice index 
Y = 120  # coronal slice index 
X = 40   # sagittal slice index 

FORCE_DEVICE = None  

# ---------- Utils ----------
def load_nifti_3d(path: str):
    """Load 3D NIfTI volume and return array + affine."""
    nii = nib.load(path)
    arr = nii.get_fdata(caching='unchanged')
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.asarray(arr), nii.affine

def zscore(x: np.ndarray, eps=1e-8):
    """Z-score normalize array."""
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def pad_or_crop_center(vol: np.ndarray, target, pad_val=0):
    """Pad/crop 3D volume to target shape around center."""
    tz, ty, tx = target
    z, y, x = vol.shape
    # pad to at least target
    pz = max(tz - z, 0); py = max(ty - y, 0); px = max(tx - x, 0)
    if pz or py or px:
        vol = np.pad(vol,
                     ((pz//2, pz - pz//2),
                      (py//2, py - py//2),
                      (px//2, px - px//2)),
                     mode='constant', constant_values=pad_val)
    # crop center
    z, y, x = vol.shape
    cz = (z - tz) // 2; cy = (y - ty) // 2; cx = (x - tx) // 2
    return vol[cz:cz+tz, cy:cy+ty, cx:cx+tx]

def overlay(ax, base, mask=None, alpha=0.4, title=None):
    """Show grayscale image with optional mask overlay."""
    ax.imshow(base, cmap='gray')
    if mask is not None:
        ax.imshow(mask, alpha=alpha, interpolation='nearest')
    if title: ax.set_title(title)
    ax.axis('off')

# ---------- Device ----------
device = torch.device(FORCE_DEVICE) if FORCE_DEVICE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Device: {device}")

# ---------- Load checkpoint ----------
ckpt = torch.load(CKPT_PATH, map_location=device)
num_classes = ckpt.get('num_classes')
base_ch     = ckpt.get('base_ch', 32)
patch_size  = ckpt.get('patch_size', (256,256,128))

# ===== Small robustness: infer num_classes if not stored =====
if num_classes is None:
    sd = ckpt['state_dict']
    if 'head.weight' in sd:
        num_classes = sd['head.weight'].shape[0]
    elif 'module.head.weight' in sd:  # in case it was saved with DataParallel
        num_classes = sd['module.head.weight'].shape[0]
    else:
        raise ValueError("num_classes not found in checkpoint; cannot infer output channels.")

# ---------- Build & load model ----------
model = ImprovedUNet3D(in_ch=1, n_classes=num_classes, base=base_ch, dropout=0.1).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# ---------- Load volumes ----------
img_np, _ = load_nifti_3d(IMAGE_PATH)
img_np = zscore(img_np)
img_v = pad_or_crop_center(img_np, patch_size)

lab_v = None
if LABEL_PATH is not None and os.path.exists(LABEL_PATH):
    lab_np, _ = load_nifti_3d(LABEL_PATH)
    lab_v = pad_or_crop_center(lab_np.astype(int), patch_size)

# ---------- Inference ----------
with torch.no_grad():
    x_t = torch.from_numpy(img_v[None, None].astype(np.float32)).to(device)
    logits = model(x_t)
    pred_np = torch.argmax(logits, dim=1)[0].cpu().numpy()

# ---------- Pick slices ----------
D,H,W = img_v.shape
z = Z if Z is not None else D//2
y = Y if Y is not None else H//2
x = X if X is not None else W//2

# ---------- Plot ----------
os.makedirs("samples", exist_ok=True)  # ensure folder exists

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"{os.path.basename(IMAGE_PATH)} — z={z}, y={y}, x={x}", fontsize=14)

# Axial
overlay(axes[0,0], img_v[z], lab_v[z] if lab_v is not None else None, title=f"Axial z={z} (Label)")
overlay(axes[1,0], img_v[z], pred_np[z], title="Axial (Pred)")

# Coronal
overlay(axes[0,1], img_v[:,y,:], lab_v[:,y,:] if lab_v is not None else None, title=f"Coronal y={y} (Label)")
overlay(axes[1,1], img_v[:,y,:], pred_np[:,y,:], title="Coronal (Pred)")

# Sagittal
overlay(axes[0,2], img_v[:,:,x], lab_v[:,:,x] if lab_v is not None else None, title=f"Sagittal x={x} (Label)")
overlay(axes[1,2], img_v[:,:,x], pred_np[:,:,x], title="Sagittal (Pred)")

plt.tight_layout()

# ---------- SAVE the figure ----------
save_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0] + "_viz.png"
save_path = os.path.join("samples", save_name)
plt.savefig(save_path, bbox_inches='tight', dpi=200)
print(f"[INFO] Saved visualization to: {save_path}")
