import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F  # only for one_hot if you ever print/inspect

# Hyperparameters that affect transforms
image_size = 128
num_classes = 3  # 0: background, 1: GM, 2: WM

# Transforms for input and segmentation mask (same as yours)
img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

seg_transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
    transforms.PILToTensor(),  # preserves integer label values
])

class OASISSegDataset(Dataset):
    """
    Dataset for OASIS MR slices and segmentation masks.
    Assumes image and mask PNGs have matching filenames.
    """
    def __init__(self, img_dir, seg_dir, img_transform=img_transform, seg_transform=seg_transform):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.seg_paths = sorted(glob.glob(os.path.join(seg_dir, "*.png")))
        assert len(self.img_paths) == len(self.seg_paths), "Mismatch in image/mask count"
        self.img_transform = img_transform
        self.seg_transform = seg_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        seg = Image.open(self.seg_paths[idx])
        if self.img_transform:
            img = self.img_transform(img)
        if self.seg_transform:
            seg = self.seg_transform(seg).long()  # shape: [1, H, W]
        seg = seg.squeeze(0)  # shape: [H, W]
        if idx == 0:
            print("Unique mask values:", torch.unique(seg))
        # Explicit mapping for OASIS mask values
        segmap = torch.zeros_like(seg)
        segmap[seg == 0] = 0
        segmap[seg == 85] = 1
        segmap[seg == 170] = 2
        segmap[seg == 255] = 0  # treat ignore as background
        seg_onehot = F.one_hot(segmap, num_classes=num_classes).permute(2, 0, 1).float()
        return img, seg_onehot
