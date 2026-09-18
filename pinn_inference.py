
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio


class FloodDepthPINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        return F.softplus(self.net(x))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "pinn_flood_depth_final.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_pinn():
    model = FloodDepthPINN().to(DEVICE)

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state)
    model.eval()

    return model


def predict_depth(
    flood_probability,
    dem_path,
    rainfall_mm,
    valid_mask
):
    model = load_pinn()

    # Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)

    # Normalize DEM
    dem_valid = np.isfinite(dem)

    if dem_valid.any():
        dem_min = dem[dem_valid].min()
        dem_max = dem[dem_valid].max()

        if dem_max > dem_min:
            dem_norm = np.zeros_like(dem)
            dem_norm[dem_valid] = (
                (dem[dem_valid] - dem_min)
                / (dem_max - dem_min)
            )
        else:
            dem_norm = np.zeros_like(dem)
    else:
        dem_norm = np.zeros_like(dem)

    # Clean flood probability
    flood_probability = np.nan_to_num(
        flood_probability,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    flood_probability = np.clip(
        flood_probability,
        0.0,
        1.0
    )

    # Rainfall normalized
    rainfall_norm = float(rainfall_mm) / 100.0
    rainfall_norm = np.clip(
        rainfall_norm,
        0.0,
        1.0
    )

    rainfall_map = np.full_like(
        flood_probability,
        rainfall_norm,
        dtype=np.float32
    )

    # Create input tensor
    input_data = np.stack(
        [
            flood_probability.astype(np.float32),
            dem_norm.astype(np.float32),
            rainfall_map
        ],
        axis=0
    )

    input_tensor = torch.from_numpy(
        input_data
    ).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        depth = model(input_tensor)

    depth = depth.squeeze().cpu().numpy()

    # Apply valid/flood mask
    valid_mask = valid_mask.astype(bool)

    flood_area = (
        (flood_probability >= 0.5)
        & valid_mask
    )

    depth[~flood_area] = 0.0

    # Safety clipping
    depth = np.clip(
        depth,
        0.0,
        None
    )

    return depth
