import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 3

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2 with residual"""
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.proj  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out
# It helps extract features and refine them by performing multiple layers of convolutions.

class AttentionGate(nn.Module):
    """
    Gating on skip features 'x' with decoder features 'g' (channel-wise + spatial).
    """
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(inter_ch)
        self.act     = nn.ReLU(inplace=True)
        self.psi     = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.sig     = nn.Sigmoid()

    def forward(self, x, g):
        q = self.theta_x(x)
        k = self.phi_g(g)
        a = self.sig(self.psi(self.act(self.bn(q + k))))
        return x * a


class UNet(nn.Module):
    # setting up acrcitecture nothing happens here.
    def __init__(self, n_channels=1, n_classes=num_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # the encoder down sampling but more feature channels
        # Feture channel learned stuff about pixels in the image
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # most number of feature chanels is maximised
        self.center = DoubleConv(512, 1024)

        # decoding upsampling but with the colours using feature channels
        # deconvulition oppotiste of convultion. turning inputu with filter into a grid.
        # uses the feature channels to do so.
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.att4 = AttentionGate(512, 512, 256)  # NEW

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.att3 = AttentionGate(256, 256, 128)  # NEW

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.att2 = AttentionGate(128, 128, 64)   # NEW

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.att1 = AttentionGate(64, 64, 32)     # NEW

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    # passes the input data through the network.
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.maxpool(d1))
        d3 = self.down3(self.maxpool(d2)) #down exampling
        d4 = self.down4(self.maxpool(d3))

        center = self.center(self.maxpool(d4)) # max feature channels 

        u4 = self.up4(center)
        u4 = torch.cat([u4, self.att4(d4, u4)], dim=1)  # CHANGED: gated skip
        dec4 = self.dec4(u4)

        u3 = self.up3(dec4)
        u3 = torch.cat([u3, self.att3(d3, u3)], dim=1)  # CHANGED: gated skip
        dec3 = self.dec3(u3)

        u2 = self.up2(dec3)
        u2 = torch.cat([u2, self.att2(d2, u2)], dim=1)  # CHANGED: gated skip
        dec2 = self.dec2(u2)

        u1 = self.up1(dec2)
        u1 = torch.cat([u1, self.att1(d1, u1)], dim=1)  # CHANGED: gated skip
        dec1 = self.dec1(u1)

        out = self.final(dec1)
        # Output: [batch, num_classes, H, W] (logits)
        return out

# Loss & Metrics metric used to measure the overlap between two sets
def dice_score(pred, target, epsilon=1e-6):
    """
    Computes Dice Similarity Coefficient (DSC) per class
    pred, target: [batch, num_classes, H, W], float, binary (one-hot)
    Returns: [num_classes] array of DSC
    """
    pred = pred.float().detach() # gets vvalues of predicted
    target = target.float().detach() # gets value of target
    dscs = []
    for c in range(pred.shape[1]):
        p = pred[:, c].reshape(-1)
        t = target[:, c].reshape(-1)
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        dsc = (2. * intersection + epsilon) / (union + epsilon)
        dscs.append(dsc.item())
    return np.array(dscs)
