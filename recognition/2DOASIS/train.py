import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

from dataset import OASISSegDataset, img_transform, seg_transform, image_size, num_classes
from modules import UNet, dice_score

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")

# Hyperparameters
batch_size = 16 # 16 images at a time
num_epochs = 30
learning_rate = 1e-6

# Data directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(SCRIPT_DIR, "OASIS")
train_img_dir = os.path.join(data_root, "keras_png_slices_train")
train_seg_dir = os.path.join(data_root, "keras_png_slices_seg_train")
val_img_dir = os.path.join(data_root, "keras_png_slices_validate")
val_seg_dir = os.path.join(data_root, "keras_png_slices_seg_validate")
test_img_dir = os.path.join(data_root, "keras_png_slices_test")
test_seg_dir = os.path.join(data_root, "keras_png_slices_seg_test")

trainset = OASISSegDataset(train_img_dir, train_seg_dir, img_transform, seg_transform)
valset   = OASISSegDataset(val_img_dir,   val_seg_dir,   img_transform, seg_transform)
testset  = OASISSegDataset(test_img_dir,  test_seg_dir,  img_transform, seg_transform)

# (moved earlier so they're defined before use)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(valset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False)

# UNet Model
model = UNet(n_channels=1, n_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # expects [batch, C, H, W] logits, target [batch, C, H, W] one-hot

print("Training UNet for segmentation...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_dsc = np.zeros(num_classes)
    for imgs, segs in train_loader:
        imgs = imgs.to(device)
        segs = segs.to(device)
        optimizer.zero_grad()
        # Forward
        logits = model(imgs)  # [B, C, H, W]
        segs_argmax = torch.argmax(segs, dim=1)  # [B, H, W] for loss
        loss = criterion(logits, segs_argmax)
        # back propagation to update the model paramters
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
        # Metrics
        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
        train_dsc += dice_score(preds_onehot, segs) * imgs.size(0)
    train_loss /= len(trainset)
    train_dsc /= len(trainset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, DSC: {train_dsc}")

    # Validation
    model.eval()
    val_loss = 0
    val_dsc = np.zeros(num_classes)
    with torch.no_grad():
        for imgs, segs in val_loader:
            imgs = imgs.to(device)
            segs = segs.to(device)
            logits = model(imgs)
            segs_argmax = torch.argmax(segs, dim=1)
            loss = criterion(logits, segs_argmax)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
            val_dsc += dice_score(preds_onehot, segs) * imgs.size(0)
        val_loss /= len(valset)
        val_dsc /= len(valset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, DSC: {val_dsc}")

# Inference & DSC on Test Set
model.eval()
test_dsc = np.zeros(num_classes)
with torch.no_grad():
    for imgs, segs in test_loader: # test data iterate
        imgs = imgs.to(device) # grabs img
        segs = segs.to(device) # grabs seg image
        logits = model(imgs) # img pass though model
        preds = torch.argmax(logits, dim=1) 
        # compares the result to the seg
        preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
        test_dsc += dice_score(preds_onehot, segs) * imgs.size(0)
    test_dsc /= len(testset)
print(f"Test DSC (per class): {test_dsc}")

# Visualize a few Segmentation Results
def plot_segmentation(img, seg_true, seg_pred, num_classes=num_classes):
    """
    Plot image, true mask, predicted mask for all classes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.squeeze(), cmap='gray')
    axes[0].set_title('Input Image')
    axes[1].imshow(torch.argmax(seg_true, dim=0).cpu(), cmap='tab20')
    axes[1].set_title('True Segmentation')
    axes[2].imshow(torch.argmax(seg_pred, dim=0).cpu(), cmap='tab20')
    axes[2].set_title('Predicted Segmentation')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

print("Showing sample segmentation results...")
model.eval()
with torch.no_grad():
    imgs, segs = next(iter(test_loader))
    imgs = imgs.to(device)
    segs = segs.to(device)
    logits = model(imgs)
    preds = torch.argmax(logits, dim=1)
    preds_onehot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    for i in range(min(5, imgs.size(0))):
        plot_segmentation(imgs[i].cpu(), segs[i].cpu(), preds_onehot[i].cpu(), num_classes=num_classes)

# (one extra line so predict.py can run standalone)
torch.save(model.state_dict(), "unet_weights.pth")
