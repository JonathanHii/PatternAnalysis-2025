import os, glob
from typing import List, Tuple
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# ----------------------------
# I/O + preprocessing utils
# ----------------------------
def find_pairs(img_dir: str, lab_dir: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (key, img_path, lab_path)
    key: e.g., 'B006_Week0' derived from *_LFOV / *_SEMANTIC
    """
    def key_from(path: str):
        base = os.path.basename(path)
        base = base.replace("_LFOV.nii.gz", "").replace("_SEMANTIC.nii.gz", "")
        base = base.replace(".nii.gz", "").replace(".nii", "")
        return base

    imgs = sorted(glob.glob(os.path.join(img_dir, "*_LFOV.nii*")))
    labs = sorted(glob.glob(os.path.join(lab_dir, "*_SEMANTIC.nii*")))
    img_map = {key_from(p): p for p in imgs}
    lab_map = {key_from(p): p for p in labs}
    keys = sorted(set(img_map.keys()) & set(lab_map.keys()))
    pairs = [(k, img_map[k], lab_map[k]) for k in keys]
    if len(pairs) == 0:
        raise RuntimeError("No matching LFOV/SEMANTIC pairs found.")
    print(f"[INFO] Found {len(pairs)} 3D pairs.")
    return pairs

def load_nifti_3d(path: str):
    """Return (vol, affine). Squeezes trailing singleton dims."""
    nii = nib.load(path)
    arr = nii.get_fdata(caching='unchanged')
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    return arr, nii.affine

def zscore(x: np.ndarray, eps=1e-8):
    m, s = x.mean(), x.std()
    return (x - m) / (s + eps)

def pad_or_crop_center(vol: np.ndarray, target, pad_val=0) -> np.ndarray:
    """Center pad/crop to target shape."""
    z, y, x = vol.shape
    tz, ty, tx = target
    # pad to at least target
    pz = max(tz - z, 0); py = max(ty - y, 0); px = max(tx - x, 0)
    if pz or py or px:
        vol = np.pad(
            vol,
            ((pz//2, pz - pz//2),
             (py//2, py - py//2),
             (px//2, px - px//2)),
            mode='constant', constant_values=pad_val
        )
    # then crop center
    z, y, x = vol.shape
    cz = (z - tz) // 2; cy = (y - ty) // 2; cx = (x - tx) // 2
    vol = vol[cz:cz+tz, cy:cy+ty, cx:cx+tx]
    return vol

def dice_per_class(pred_oh: torch.Tensor, tgt_oh: torch.Tensor, eps=1e-6):
    """
    Compute per-class Dice, ignoring classes that don't appear in target or prediction.
    Returns NaN for classes that are completely absent.
    """
    p = pred_oh.float().detach()
    t = tgt_oh.float().detach()
    inter = (p * t).sum(dim=(0, 2, 3, 4))
    denom = p.sum(dim=(0, 2, 3, 4)) + t.sum(dim=(0, 2, 3, 4))
    valid = denom > 0
    dsc = torch.full_like(inter, float('nan'))
    dsc[valid] = (2 * inter[valid] + eps) / (denom[valid] + eps)
    return dsc.cpu().numpy()

# ----------------------------
# Dataset
# ----------------------------
class HipMRI3DDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], patch_size=(128,128,128), norm=True):
        self.items = pairs
        self.ps = patch_size
        self.norm = norm

        # infer num_classes by scanning a few labels
        uniq = set()
        for _, _, lp in self.items[:min(6, len(self.items))]:
            arr, _ = load_nifti_3d(lp)
            uniq.update(np.unique(arr).astype(int).tolist())
        self.num_classes = int(max(uniq)) + 1
        print(f"[INFO] Detected num_classes={self.num_classes} (from sample labels).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        key, ip, lp = self.items[idx]
        img, _ = load_nifti_3d(ip)
        lab, _ = load_nifti_3d(lp)

        # basic sanity
        if img.shape != lab.shape:
            minz = min(img.shape[0], lab.shape[0])
            miny = min(img.shape[1], lab.shape[1])
            minx = min(img.shape[2], lab.shape[2])
            img = img[:minz, :miny, :minx]
            lab = lab[:minz, :miny, :minx]

        if self.norm:
            img = zscore(img)
        img = pad_or_crop_center(img, self.ps, pad_val=0)
        lab = pad_or_crop_center(lab, self.ps, pad_val=0)

        img_t = torch.from_numpy(img.astype(np.float32))[None, ...]   # (1,D,H,W)
        lab_t = torch.from_numpy(lab.astype(np.int64))                # (D,H,W)
        return img_t, lab_t, key
