import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Building blocks
# ----------------------------
class SE3D(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d(1)
        mid = max(1, ch // r)
        self.fc = nn.Sequential(
            nn.Conv3d(ch, mid, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid, ch, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class Res3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0, groups=16, use_se=True):
        super().__init__()
        self.c1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.g1 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.a1 = nn.SiLU(inplace=True)
        self.dp = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.c2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.g2 = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.se = SE3D(out_ch) if use_se else nn.Identity()
        self.a2 = nn.SiLU(inplace=True)
    def forward(self, x):
        idt = self.proj(x)
        x = self.c1(x); x = self.g1(x); x = self.a1(x)
        x = self.dp(x)
        x = self.c2(x); x = self.g2(x)
        x = x + idt
        x = self.se(x)
        x = self.a2(x)
        return x

class ASPP3D(nn.Module):
    def __init__(self, in_ch, out_ch, atrous=(1, 2, 4, 6)):
        super().__init__()
        self.br = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
                nn.SiLU(inplace=True)
            )
        ])
        for d in atrous:
            self.br.append(nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
                nn.SiLU(inplace=True)
            ))
        self.imgpool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.SiLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv3d(out_ch * (len(self.br) + 1), out_ch, 1, bias=False),
            nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        D, H, W = x.shape[-3:]
        feats = [b(x) for b in self.br]
        g = self.imgpool(x)
        g = F.interpolate(g, size=(D, H, W), mode='trilinear', align_corners=False)
        feats.append(g)
        x = torch.cat(feats, dim=1)
        return self.proj(x)

class AttnGate3D(nn.Module):
    def __init__(self, in_skip, in_g, inter):
        super().__init__()
        self.theta = nn.Conv3d(in_skip, inter, 1, bias=False)
        self.phi   = nn.Conv3d(in_g,    inter, 1, bias=False)
        self.psi   = nn.Conv3d(inter,   1,     1, bias=True)
        self.act   = nn.SiLU(inplace=True)
        self.sig   = nn.Sigmoid()
    def forward(self, x_skip, g):
        t = self.theta(x_skip)
        p = self.phi(g)
        if t.shape[-3:] != p.shape[-3:]:
            p = F.interpolate(p, size=t.shape[-3:], mode='trilinear', align_corners=False)
        a = self.sig(self.psi(self.act(t + p)))
        return x_skip * a

class Up3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, groups=16, use_se=True):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
        )
        self.attn = AttnGate3D(skip_ch, out_ch, inter=max(out_ch // 2, 1))
        self.res  = Res3D(out_ch + skip_ch, out_ch, dropout=dropout, groups=groups, use_se=use_se)
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attn(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x

# ----------------------------
# Model
# ----------------------------
class ImprovedUNet3D(nn.Module):
    def __init__(self, in_ch=1, n_classes=6, base=32, dropout=0.1, deep_supervision=False):
        super().__init__()
        C = [base, base*2, base*4, base*8, base*16]
        self.enc1 = Res3D(in_ch, C[0], dropout=0.0)
        self.enc2 = Res3D(C[0], C[1], dropout=dropout*0.25)
        self.enc3 = Res3D(C[1], C[2], dropout=dropout*0.5)
        self.enc4 = Res3D(C[2], C[3], dropout=dropout)
        self.pool = nn.MaxPool3d(2)

        self.aspp = ASPP3D(C[3], C[4]//2)
        self.bot  = Res3D(C[4]//2, C[4], dropout=dropout)

        self.up4 = Up3D(C[4], C[3], C[3], dropout=dropout)
        self.up3 = Up3D(C[3], C[2], C[2], dropout=dropout*0.5)
        self.up2 = Up3D(C[2], C[1], C[1], dropout=dropout*0.25)
        self.up1 = Up3D(C[1], C[0], C[0], dropout=0.0)

        self.head = nn.Conv3d(C[0], n_classes, 1)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux3 = nn.Conv3d(C[2], n_classes, 1)
            self.aux2 = nn.Conv3d(C[1], n_classes, 1)
            self.aux1 = nn.Conv3d(C[0], n_classes, 1)

    def forward(self, x):
        d1 = self.enc1(x)              # 1/1
        d2 = self.enc2(self.pool(d1))  # 1/2
        d3 = self.enc3(self.pool(d2))  # 1/4
        d4 = self.enc4(self.pool(d3))  # 1/8
        b  = self.aspp(self.pool(d4))  # 1/16
        b  = self.bot(b)
        u4 = self.up4(b,  d4)          # 1/8
        u3 = self.up3(u4, d3)          # 1/4
        u2 = self.up2(u3, d2)          # 1/2
        u1 = self.up1(u2, d1)          # 1/1
        logits = self.head(u1)
        if self.deep_supervision and self.training:
            a3 = F.interpolate(self.aux3(u3), size=logits.shape[-3:], mode='trilinear', align_corners=False)
            a2 = F.interpolate(self.aux2(u2), size=logits.shape[-3:], mode='trilinear', align_corners=False)
            a1 = self.aux1(u1)
            return logits + 0.3*a1 + 0.2*a2 + 0.1*a3
        return logits


