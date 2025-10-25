import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

from dataset import OASISSegDataset, img_transform, seg_transform, num_classes
from modules import UNet

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")

# Data directories
data_root = "./OASIS"
test_img_dir = os.path.join(data_root, "keras_png_slices_test")
test_seg_dir = os.path.join(data_root, "keras_png_slices_seg_test")

testset = OASISSegDataset(test_img_dir, test_seg_dir, img_transform, seg_transform)
test_loader = DataLoader(testset, batch_size=16, shuffle=False)

# Model
model = UNet(n_channels=1, n_classes=num_classes).to(device)
weights_path = "unet_weights.pth"  # change if your weights are elsewhere

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at: {weights_path}")

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Output directory
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# Plotting helper
def plot_and_save_segmentation(img, seg_true, seg_pred, idx, num_classes=num_classes):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[1].imshow(torch.argmax(seg_true, dim=0).cpu(), cmap='tab20')
    axes[1].set_title('True Segmentation')
    axes[2].imshow(torch.argmax(seg_pred, dim=0).cpu(), cmap='tab20')
    axes[2].set_title('Predicted Segmentation')

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f"segmentation_result_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # close to avoid memory issues

    print(f"Saved result to: {save_path}")

print("Generating and saving sample segmentation results...")
with torch.no_grad():
    imgs, segs = next(iter(test_loader))
    imgs = imgs.to(device)
    segs = segs.to(device)
    logits = model(imgs)
    preds = torch.argmax(logits, dim=1)
    preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    for i in range(min(20, imgs.size(0))):
        plot_and_save_segmentation(imgs[i].cpu(), segs[i].cpu(), preds_onehot[i].cpu(), idx=i)
