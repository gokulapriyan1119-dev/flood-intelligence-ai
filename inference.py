import os

import numpy as np
import torch
import torch.nn as nn
import rasterio


# ==============================
# U-NET ARCHITECTURE
# ==============================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        self.output = nn.Conv2d(
            64, out_channels, 1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# LOAD TRAINED MODEL
# ==============================

MODEL_PATH = (
    "/content/flood_project/"
    "unet_flood_segmentation.pth"
)

model = UNet(
    in_channels=2,
    out_channels=1
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ==============================
# SAR PREPROCESSING
# ==============================

def preprocess_sar(sar):

    sar = sar.astype(np.float32)

    valid = np.isfinite(sar).all(axis=0)

    processed = sar.copy()

    for b in range(sar.shape[0]):

        values = sar[b][valid]

        if len(values) > 0:

            min_val = values.min()
            max_val = values.max()

            if max_val > min_val:

                processed[b][valid] = (
                    (processed[b][valid] - min_val)
                    / (max_val - min_val)
                )

    processed[:, ~valid] = 0.0

    return processed, valid


# ==============================
# SAR FLOOD PREDICTION
# ==============================

def predict_flood(sar_path, threshold=0.5):

    with rasterio.open(sar_path) as src:

        sar = src.read()

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

    if sar.shape[0] < 2:

        raise ValueError(
            "Input SAR GeoTIFF must contain "
            "at least 2 bands: VV and VH."
        )

    # Use VV + VH
    sar = sar[:2]

    processed, valid = preprocess_sar(sar)

    tensor = torch.from_numpy(
        processed
    ).unsqueeze(0).float().to(device)

    with torch.no_grad():

        logits = model(tensor)

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    # Remove invalid SAR footprint
    probability[~valid] = 0.0

    flood_mask = (
        probability >= threshold
    ) & valid

    return {
        "probability": probability,
        "mask": flood_mask,
        "valid": valid,
        "profile": profile,
        "transform": transform,
        "crs": crs
    }


print("🔥 FLOOD INFERENCE MODULE READY")
print("Device:", device)
print("Model:", MODEL_PATH)
