
import streamlit as st
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import os
import sys
from report_generator import create_flood_report

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Geo-Spatial AI | Flood Intelligence",
    page_icon="🌊",
    layout="wide"
)


# ============================================================
# FUTURISTIC UI THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 82% 8%, rgba(20,150,255,.12), transparent 28%),
        radial-gradient(circle at 10% 92%, rgba(0,220,255,.07), transparent 25%),
        #05080d;
    color: #eef7ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #070b11;
    border-right: 1px solid rgba(90,180,255,.16);
}

[data-testid="stSidebar"] * {
    color: #e9f4fb;
}

.fi-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 18px 0;
}

.fi-logo {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#0bbcff,#1767ff);
    box-shadow: 0 0 25px rgba(11,188,255,.28);
    font-size: 23px;
}

.fi-brand-title {
    font-size: 18px;
    font-weight: 900;
    letter-spacing: .07em;
}

.fi-brand-sub {
    color: #7091a5;
    font-size: 10px;
    margin-top: 2px;
}

.fi-top {
    display: flex;
    justify-content: space-between;
    align-items: end;
    border-bottom: 1px solid rgba(90,180,255,.15);
    padding-bottom: 16px;
    margin-bottom: 18px;
}

.fi-kicker {
    color: #55cfff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .14em;
}

.fi-title {
    font-size: 32px;
    font-weight: 900;
    line-height: 1.05;
    margin-top: 7px;
}

.fi-title span {
    color: #18c7ff;
}

.fi-online {
    color: #36e69e;
    font-size: 12px;
    font-weight: 800;
}

.fi-hero {
    min-height: 150px;
    padding: 26px 30px;
    border-radius: 18px;
    border: 1px solid rgba(90,180,255,.20);
    background:
        radial-gradient(circle at 88% 45%, rgba(0,170,255,.17), transparent 27%),
        linear-gradient(115deg,#091827,#050b12 75%);
    box-shadow: inset 0 1px rgba(255,255,255,.03);
    margin-bottom: 16px;
}

.fi-hero-small {
    color: #72a5bb;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .12em;
}

.fi-hero-main {
    font-size: 29px;
    font-weight: 900;
    margin-top: 8px;
}

.fi-hero-main span {
    color: #19c8ff;
}

.fi-hero-desc {
    color: #86a7b8;
    font-size: 13px;
    margin-top: 7px;
}

.fi-card {
    min-height: 112px;
    padding: 16px;
    border-radius: 15px;
    border: 1px solid rgba(90,180,255,.18);
    background: linear-gradient(145deg,#0a1928,#050d16);
}

.fi-label {
    color: #7295a9;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .11em;
    text-transform: uppercase;
}

.fi-value {
    font-size: 27px;
    font-weight: 900;
    margin-top: 8px;
}

.fi-line {
    height: 4px;
    background: #142a3d;
    border-radius: 10px;
    margin-top: 12px;
}

.fi-line > div {
    height: 100%;
    background: linear-gradient(90deg,#0ec8ff,#116dff);
    border-radius: 10px;
}

.fi-panel {
    border: 1px solid rgba(90,180,255,.18);
    border-radius: 16px;
    background: linear-gradient(145deg,#091a2a,#050d15);
    padding: 14px;
}

.fi-section {
    color: #57ceff;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin: 16px 0 9px 0;
}

.fi-status {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    padding: 10px 13px;
    margin-top: 12px;
    border-radius: 12px;
    border: 1px solid rgba(50,230,160,.16);
    background: #07151f;
    color: #a9c3d1;
    font-size: 11px;
}

.fi-status b {
    color: #32e6a0;
}

.fi-risk {
    padding: 16px;
    border-radius: 15px;
    border: 1px solid rgba(255,170,60,.25);
    background: linear-gradient(145deg,rgba(70,48,12,.38),rgba(20,15,8,.30));
}

.fi-risk-label {
    color: #e5ad59;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.fi-risk-value {
    font-size: 28px;
    font-weight: 900;
    margin-top: 5px;
}

.fi-step {
    text-align: center;
    padding: 11px 7px;
    border: 1px solid rgba(90,180,255,.16);
    border-radius: 12px;
    background: #071522;
}

.fi-step-icon {
    font-size: 21px;
}

.fi-step-title {
    font-size: 10px;
    font-weight: 850;
    margin-top: 5px;
}

.fi-step-sub {
    color: #6e8fa3;
    font-size: 8px;
    margin-top: 2px;
}

div[data-testid="stDownloadButton"] > button {
    border-radius: 9px;
    border: 1px solid rgba(25,198,255,.28);
    background: linear-gradient(135deg,#087fff,#1457e9);
    color: white;
    font-weight: 800;
}

hr {
    border-color: rgba(90,180,255,.12);
}

.fi-footer {
    border-top: 1px solid rgba(90,180,255,.14);
    margin-top: 22px;
    padding-top: 12px;
    color: #628397;
    font-size: 10px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload inference module
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

try:
    from inference import predict_flood
except Exception:
    predict_flood = None

SAR_PATH = f"{PROJECT_DIR}/Bolivia_103757_S1Hand.tif"
PROB_PATH = f"{PROJECT_DIR}/flood_probability.tif"
DEPTH_PATH = f"{PROJECT_DIR}/flood_depth_final_corrected.tif"
RISK_PATH = f"{PROJECT_DIR}/flood_risk_map.tif"
RAINFALL_PATH = f"{PROJECT_DIR}/rainfall_24h.csv"
SUMMARY_PATH = f"{PROJECT_DIR}/project_results.json"

UNET_PATH = f"{PROJECT_DIR}/unet_flood_segmentation.pth"
PINN_PATH = f"{PROJECT_DIR}/pinn_flood_depth_final.pth"
DEM_PATH = f"{PROJECT_DIR}/dem/bolivia_dem_aligned.tif"


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.conv(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=2,
        out_channels=1
    ):

        super().__init__()

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        self.output = nn.Conv2d(
            64,
            out_channels,
            1
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

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# PINN
# ============================================================

class FloodDepthPINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                16,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                1,
                1
            )
        )

    def forward(self, x):

        return F.softplus(
            self.net(x)
        )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    unet = UNet().to(device)

    unet.load_state_dict(
        torch.load(
            UNET_PATH,
            map_location=device
        )
    )

    unet.eval()

    pinn = FloodDepthPINN().to(device)

    pinn.load_state_dict(
        torch.load(
            PINN_PATH,
            map_location=device
        )
    )

    pinn.eval()

    return unet, pinn


# Load both trained models once for the Streamlit session.
unet, pinn = load_models()


# ============================================================
# LOAD RASTER
# ============================================================

def read_raster(path):

    with rasterio.open(path) as src:

        data = src.read()

        profile = src.profile

    return data, profile


# ============================================================
# LOAD DEM
# ============================================================

def load_dem():

    with rasterio.open(
        DEM_PATH
    ) as src:

        dem = src.read(1).astype(
            np.float32
        )

    dem_min = np.nanmin(dem)

    dem_max = np.nanmax(dem)

    dem = (
        dem - dem_min
    ) / (
        dem_max - dem_min + 1e-8
    )

    dem = np.nan_to_num(
        dem
    )

    return dem


# ============================================================
# LOAD DEMO DATA
# ============================================================

@st.cache_data
def load_demo_data():

    probability = read_raster(
        PROB_PATH
    )[0][0]

    depth = read_raster(
        DEPTH_PATH
    )[0][0]

    risk = read_raster(
        RISK_PATH
    )[0][0]

    sar = read_raster(
        SAR_PATH
    )[0]

    return sar, probability, depth, risk


# ============================================================
# RAINFALL
# ============================================================

@st.cache_data
def load_rainfall():

    df = pd.read_csv(
        RAINFALL_PATH
    )

    rain_col = [
        c for c in df.columns
        if (
            "rain" in c.lower()
            or
            "precip" in c.lower()
        )
    ][0]

    rainfall = df[
        rain_col
    ].astype(float)

    return rainfall


# ============================================================
# PINN HELPERS
# ============================================================

def normalize_dem_array(dem):
    dem = dem.astype(np.float32)
    valid = np.isfinite(dem)
    if valid.sum() == 0:
        return np.zeros_like(dem, dtype=np.float32)
    values = dem[valid]
    dem_min = values.min()
    dem_max = values.max()
    if dem_max > dem_min:
        dem = (dem - dem_min) / (dem_max - dem_min)
    else:
        dem = np.zeros_like(dem, dtype=np.float32)
    dem[~valid] = 0.0
    return dem.astype(np.float32)


def make_geotiff_bytes(array, profile, dtype, nodata=0):
    out_profile = profile.copy()
    out_profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        tiled=False
    )
    with MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(array.astype(dtype), 1)
        return memfile.read()


def calculate_risk_map(depth, flood_area):
    risk_map = np.zeros_like(depth, dtype=np.uint8)
    risk_map[flood_area & (depth < 0.10)] = 1
    risk_map[flood_area & (depth >= 0.10) & (depth < 0.25)] = 2
    risk_map[flood_area & (depth >= 0.25) & (depth < 0.40)] = 3
    risk_map[flood_area & (depth >= 0.40)] = 4
    return risk_map





# ============================================================
# FINAL UI THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 78% 8%, rgba(0,150,255,.11), transparent 28%),
        radial-gradient(circle at 10% 90%, rgba(0,220,255,.06), transparent 25%),
        #05080d;
    color:#eef7ff;
}
.block-container {max-width:1450px;padding-top:1rem;}
[data-testid="stSidebar"] {background:#070b11;border-right:1px solid rgba(80,180,255,.16);}
[data-testid="stSidebar"] * {color:#eaf5fb;}
.fi-brand{display:flex;align-items:center;gap:12px;padding:6px 0 20px;}
.fi-logo{width:44px;height:44px;border-radius:13px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0bbcff,#1767ff);box-shadow:0 0 25px rgba(11,188,255,.3);font-size:24px;}
.fi-brand-title{font-size:18px;font-weight:900;letter-spacing:.07em;}
.fi-brand-sub{color:#7094aa;font-size:10px;margin-top:3px;}
.fi-top{display:flex;justify-content:space-between;align-items:end;border-bottom:1px solid rgba(80,180,255,.16);padding-bottom:15px;margin-bottom:18px;}
.fi-kicker{color:#55cfff;font-size:11px;font-weight:800;letter-spacing:.14em;}
.fi-title{font-size:30px;font-weight:900;line-height:1.05;margin-top:6px;}
.fi-title span{color:#19c8ff;}
.fi-online{color:#35e59e;font-size:12px;font-weight:800;}
.fi-hero{min-height:135px;padding:25px 29px;border-radius:18px;border:1px solid rgba(80,180,255,.2);background:radial-gradient(circle at 88% 45%,rgba(0,170,255,.18),transparent 27%),linear-gradient(115deg,#091827,#050b12 75%);box-shadow:inset 0 1px rgba(255,255,255,.03);margin-bottom:16px;}
.fi-hero-small{color:#70bddd;font-size:10px;font-weight:800;letter-spacing:.13em;}
.fi-hero-main{font-size:29px;font-weight:900;margin-top:8px;}
.fi-hero-main span{color:#19c8ff;}
.fi-hero-desc{color:#86a7b8;font-size:13px;margin-top:7px;}
.fi-card{min-height:105px;padding:15px;border-radius:15px;border:1px solid rgba(80,180,255,.18);background:linear-gradient(145deg,#0a1928,#050d16);}
.fi-label{color:#7295a9;font-size:10px;font-weight:800;letter-spacing:.11em;text-transform:uppercase;}
.fi-value{font-size:27px;font-weight:900;margin-top:7px;}
.fi-line{height:4px;background:#142a3d;border-radius:10px;margin-top:12px;}
.fi-line>div{height:100%;background:linear-gradient(90deg,#0ec8ff,#116dff);border-radius:10px;}
.fi-panel{border:1px solid rgba(80,180,255,.18);border-radius:16px;background:linear-gradient(145deg,#091a2a,#050d15);padding:14px;}
.fi-section{color:#57ceff;font-size:11px;font-weight:900;letter-spacing:.13em;text-transform:uppercase;margin:17px 0 9px;}
.fi-status{display:flex;gap:18px;flex-wrap:wrap;padding:10px 13px;margin-top:12px;border-radius:12px;border:1px solid rgba(50,230,160,.16);background:#07151f;color:#a9c3d1;font-size:11px;}
.fi-status b{color:#32e6a0;}
.fi-risk{padding:16px;border-radius:15px;border:1px solid rgba(255,170,60,.25);background:linear-gradient(145deg,rgba(70,48,12,.38),rgba(20,15,8,.3));margin-bottom:12px;}
.fi-risk-label{color:#e5ad59;font-size:10px;font-weight:900;letter-spacing:.12em;}
.fi-risk-value{font-size:28px;font-weight:900;margin-top:5px;}
.fi-step{text-align:center;padding:11px 5px;border:1px solid rgba(80,180,255,.16);border-radius:12px;background:#071522;}
.fi-step-icon{font-size:20px}.fi-step-title{font-size:10px;font-weight:850;margin-top:5px}.fi-step-sub{color:#6e8fa3;font-size:8px;margin-top:2px;}
div[data-testid="stDownloadButton"]>button{border-radius:9px;border:1px solid rgba(25,198,255,.28);background:linear-gradient(135deg,#087fff,#1457e9);color:white;font-weight:800;}
.fi-footer{border-top:1px solid rgba(80,180,255,.14);margin-top:22px;padding-top:12px;color:#628397;font-size:10px;}
</style>
""", unsafe_allow_html=True)

# Top navigation
st.markdown(
    """
    <div class="fi-top">
        <div>
            <div class="fi-kicker">GEO-SPATIAL AI • FLOOD INTELLIGENCE</div>
            <div class="fi-title">THE POWER OF <span>SATELLITE DATA & AI</span></div>
        </div>
        <div class="fi-online">● SYSTEM ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="fi-brand">
            <div class="fi-logo">🌊</div>
            <div>
                <div class="fi-brand-title">FLOOD INTELLIGENCE</div>
                <div class="fi-brand-sub">GEO-SPATIAL AI • FLOOD ANALYTICS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**CONTROL CENTER**")
    analysis_mode = st.radio(
        "Navigation",
        ["Demo Analysis", "Upload Analysis", "Model Information"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**DATA SOURCES**")
    st.caption("🛰️ Sentinel-1 SAR")
    st.caption("⛰️ Copernicus GLO-30 DEM")
    st.caption("🌧️ 24h Rainfall")

    st.divider()
    st.markdown("**AI MODELS**")
    st.caption("✓ U-Net Segmentation")
    st.caption("✓ PINN Depth Estimation")
    st.caption("✓ Risk Classification")



# ============================================================
# UPLOAD ANALYSIS
# ============================================================

if analysis_mode == "Upload Analysis":

    st.header("📤 Upload Sentinel-1 SAR")
    st.write(
        "Upload a Sentinel-1 SAR GeoTIFF containing VV and VH bands "
        "to run the trained U-Net and, for the matching research scene, "
        "the PINN depth and risk estimation pipeline."
    )

    st.info(
        "Supported input: GeoTIFF with at least 2 SAR bands (VV + VH). "
        "PINN depth is demonstrated with the matching Bolivia study scene "
        "because its DEM and rainfall inputs are scene-specific."
    )

    uploaded_file = st.file_uploader(
        "Choose Sentinel-1 SAR GeoTIFF",
        type=["tif", "tiff"]
    )

    if uploaded_file is not None:

        upload_dir = os.path.join(PROJECT_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, uploaded_file.name)

        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if predict_flood is None:
            st.error(
                "Inference module could not be loaded. "
                "Make sure inference.py exists inside flood_project."
            )
            st.stop()

        try:
            # --------------------------------------------------------
            # STEP 1 — U-NET FLOOD SEGMENTATION
            # --------------------------------------------------------
            with st.spinner("🧠 Running U-Net flood segmentation..."):
                result = predict_flood(upload_path)

            probability = result["probability"]
            flood_mask = result["mask"]
            valid_mask = result["valid"]

            with rasterio.open(upload_path) as src:
                sar_data = src.read(1).astype(np.float32)
                upload_profile = src.profile.copy()
                upload_crs = src.crs
                upload_bounds = src.bounds

            valid_pixels = int(valid_mask.sum())
            flood_pixels = int(flood_mask.sum())
            flood_percentage = (
                flood_pixels / valid_pixels * 100
                if valid_pixels > 0 else 0
            )

            st.success("✅ U-Net flood segmentation completed!")

            c1, c2, c3 = st.columns(3)
            c1.metric("Valid SAR Pixels", f"{valid_pixels:,}")
            c2.metric("Flood Pixels", f"{flood_pixels:,}")
            c3.metric("Flood Coverage", f"{flood_percentage:.2f}%")

            st.divider()

            # --------------------------------------------------------
            # U-NET MAPS
            # --------------------------------------------------------
            st.subheader("🛰️ U-Net Flood Detection")
            m1, m2 = st.columns(2)

            with m1:
                st.markdown("### Sentinel-1 SAR")
                sar_display = sar_data.copy()
                sar_display[~valid_mask] = np.nan
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.imshow(sar_display, cmap="gray")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with m2:
                st.markdown("### Flood Probability")
                prob_display = probability.copy()
                prob_display[~valid_mask] = np.nan
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(prob_display, cmap="viridis", vmin=0, vmax=1)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 5))
            mask_display = np.ma.masked_where(~flood_mask, flood_mask)
            ax.imshow(mask_display, cmap="Blues")
            ax.axis("off")
            ax.set_title("U-Net Flood Mask")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # --------------------------------------------------------
            # STEP 2 — CHECK SCENE MATCH FOR PINN
            # --------------------------------------------------------
            pinn_ready = False
            with rasterio.open(DEM_PATH) as dem_src:
                dem_shape = (dem_src.height, dem_src.width)
                dem_crs = dem_src.crs
                dem_bounds = dem_src.bounds

            same_shape = probability.shape == dem_shape
            same_crs = upload_crs == dem_crs
            same_bounds = all(
                abs(a - b) < 1e-4
                for a, b in zip(upload_bounds, dem_bounds)
            )
            pinn_ready = same_shape and same_crs and same_bounds

            if not pinn_ready:
                st.warning(
                    "⚠️ U-Net analysis is complete, but PINN depth was not "
                    "run because the uploaded SAR scene does not match the "
                    "scene-specific DEM used to train the current PINN. "
                    "This prevents applying the Bolivia DEM/rainfall to an "
                    "unrelated location."
                )

            else:
                # ----------------------------------------------------
                # STEP 3 — PINN DEPTH ESTIMATION
                # ----------------------------------------------------
                with st.spinner("🌊 Running PINN flood-depth estimation..."):
                    dem = load_dem()
                    rainfall = load_rainfall()
                    total_rainfall = float(rainfall.sum())

                    probability_tensor = torch.from_numpy(
                        probability.astype(np.float32)
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    dem_tensor = torch.from_numpy(
                        dem.astype(np.float32)
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    rainfall_map = np.full(
                        probability.shape,
                        total_rainfall / 100.0,
                        dtype=np.float32
                    )
                    rainfall_tensor = torch.from_numpy(
                        rainfall_map
                    ).unsqueeze(0).unsqueeze(0).to(device)

                    pinn_input = torch.cat(
                        [probability_tensor, dem_tensor, rainfall_tensor],
                        dim=1
                    )

                    # Load/access the PINN explicitly inside the upload path.
                    # This avoids any stale/global variable issue in Streamlit.
                    _, pinn_model = load_models()
                    with torch.no_grad():
                        depth = pinn_model(pinn_input).squeeze().cpu().numpy()

                    flood_area = (probability >= 0.5) & valid_mask
                    depth[~flood_area] = 0.0
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                    depth = np.maximum(depth, 0.0).astype(np.float32)

                    risk = calculate_risk_map(depth, flood_area)

                depth_values = depth[flood_area]
                average_depth = float(depth_values.mean()) if len(depth_values) else 0.0
                maximum_depth = float(depth_values.max()) if len(depth_values) else 0.0

                if maximum_depth < 0.10:
                    risk_level = "LOW"
                elif maximum_depth < 0.25:
                    risk_level = "MODERATE"
                elif maximum_depth < 0.40:
                    risk_level = "HIGH"
                else:
                    risk_level = "SEVERE"

                st.success("✅ PINN depth + risk estimation completed!")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Average Depth", f"{average_depth:.3f} m")
                k2.metric("Maximum Depth", f"{maximum_depth:.3f} m")
                k3.metric("24h Rainfall", f"{total_rainfall:.1f} mm")
                k4.metric("Estimated Risk", risk_level)

                st.divider()

                st.subheader("🌊 PINN Flood Depth")
                fig, ax = plt.subplots(figsize=(10, 6))
                depth_display = np.ma.masked_where(depth <= 0, depth)
                im = ax.imshow(
                    depth_display,
                    cmap="turbo",
                    vmin=0,
                    vmax=max(maximum_depth, 0.4)
                )
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.035, label="Depth (m)")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.subheader("⚠️ Flood Risk Map")
                fig, ax = plt.subplots(figsize=(10, 6))
                risk_display = np.ma.masked_where(risk == 0, risk)
                im = ax.imshow(
                    risk_display,
                    cmap="RdYlGn_r",
                    vmin=1,
                    vmax=4
                )
                ax.axis("off")
                cbar = fig.colorbar(im, ax=ax, fraction=0.035)
                cbar.set_ticks([1, 2, 3, 4])
                cbar.set_ticklabels(["Low", "Moderate", "High", "Severe"])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric("Low", f"{int((risk == 1).sum()):,}")
                rc2.metric("Moderate", f"{int((risk == 2).sum()):,}")
                rc3.metric("High", f"{int((risk == 3).sum()):,}")
                rc4.metric("Severe", f"{int((risk == 4).sum()):,}")

                # GeoTIFF downloads
                depth_bytes = make_geotiff_bytes(
                    depth, upload_profile, "float32", 0
                )
                risk_bytes = make_geotiff_bytes(
                    risk, upload_profile, "uint8", 0
                )
                probability_bytes = make_geotiff_bytes(
                    probability, upload_profile, "float32", 0
                )

                st.subheader("📥 Download Analysis Results")
                d1, d2, d3 = st.columns(3)
                d1.download_button(
                    "🌊 Download Depth Map",
                    depth_bytes,
                    file_name="flood_depth_upload.tif",
                    mime="image/tiff"
                )
                d2.download_button(
                    "⚠️ Download Risk Map",
                    risk_bytes,
                    file_name="flood_risk_upload.tif",
                    mime="image/tiff"
                )
                d3.download_button(
                    "🧠 Download Flood Probability",
                    probability_bytes,
                    file_name="flood_probability_upload.tif",
                    mime="image/tiff"
                )

                st.warning(
                    "⚠️ Depth and risk are model-estimated research outputs. "
                    "They are not validated against measured flood-depth ground truth "
                    "and should not be treated as official emergency warnings."
                )

        except Exception as e:
            st.error(f"❌ Unable to process this SAR file: {e}")

    st.stop()




# ============================================================
# MODEL INFORMATION
# ============================================================

if analysis_mode == "Model Information":

    st.markdown(
        """
        <div class="fi-hero">
            <div class="fi-hero-small">SYSTEM ARCHITECTURE</div>
            <div class="fi-hero-main">THE INTELLIGENCE <span>BEHIND THE MAP</span></div>
            <div class="fi-hero-desc">
                A multi-stage geospatial AI pipeline combining satellite observations,
                terrain, rainfall and physics-informed deep learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(5)
    info = [
        ("SATELLITE", "Sentinel-1"),
        ("SEGMENTATION", "U-Net"),
        ("TERRAIN", "DEM"),
        ("PHYSICS", "PINN"),
        ("RAINFALL", "24 Hour"),
    ]

    for col, (label, value) in zip(cols, info):
        with col:
            st.markdown(
                f"""
                <div class="fi-card">
                    <div class="fi-label">{label}</div>
                    <div class="fi-value" style="font-size:20px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="fi-section">AI PROCESSING PIPELINE</div>', unsafe_allow_html=True)

    pcols = st.columns(7)
    steps = [
        ("🛰️", "Sentinel-1", "SAR"),
        ("⚙️", "Preprocess", "VV + VH"),
        ("🧠", "U-Net", "Segmentation"),
        ("🌊", "Probability", "Flood"),
        ("⛰️", "DEM + Rain", "Conditioning"),
        ("⚛️", "PINN", "Physics"),
        ("💧", "Depth + Risk", "Output"),
    ]

    for col, (icon, title, sub) in zip(pcols, steps):
        with col:
            st.markdown(
                f"""
                <div class="fi-step">
                    <div class="fi-step-icon">{icon}</div>
                    <div class="fi-step-title">{title}</div>
                    <div class="fi-step-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="fi-section">MODEL STATUS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fi-status">
            <span>U-NET <b>● LOADED</b></span>
            <span>PINN <b>● LOADED</b></span>
            <span>DEM <b>● READY</b></span>
            <span>RAINFALL <b>● READY</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Depth and risk values are model-estimated research outputs. "
        "The current PINN prototype is scene-specific and is not a validated "
        "operational flood-warning system."
    )

    st.stop()


# ============================================================
# DEMO DASHBOARD
# ============================================================

# Demo data and metrics
# ============================================================
# DEMO ANALYSIS
# ============================================================

sar, probability, depth, risk = load_demo_data()

rainfall = load_rainfall()

total_rainfall = rainfall.sum()

max_hourly_rainfall = rainfall.max()

flood_mask = probability >= 0.5

flood_pixels = flood_mask.sum()

flood_percentage = (
    flood_pixels /
    probability.size
) * 100

flood_depth_values = depth[
    flood_mask
]

average_depth = (
    flood_depth_values.mean()
    if len(flood_depth_values) > 0
    else 0
)

maximum_depth = (
    flood_depth_values.max()
    if len(flood_depth_values) > 0
    else 0
)





risk_level = (
    "LOW" if maximum_depth < 0.10
    else "MODERATE" if maximum_depth < 0.25
    else "HIGH" if maximum_depth < 0.40
    else "SEVERE"
)

# Hero
st.markdown(
    f"""
    <div class="fi-hero">
        <div class="fi-hero-small">BOLIVIA_103757 • FLOOD EVENT ANALYSIS</div>
        <div class="fi-hero-main">THE POWER OF <span>SATELLITE DATA & AI</span></div>
        <div class="fi-hero-desc">
            From Sentinel-1 SAR detection to continuous physics-informed flood-depth estimation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI row
st.markdown('<div class="fi-section">FLOOD EVENT OVERVIEW</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
card_data = [
    ("FLOOD COVERAGE", f"{flood_percentage:.2f}%", min(flood_percentage, 100)),
    ("AVERAGE DEPTH", f"{average_depth:.3f} m", min(average_depth / .5 * 100, 100)),
    ("MAXIMUM DEPTH", f"{maximum_depth:.3f} m", min(maximum_depth / .5 * 100, 100)),
    ("24H RAINFALL", f"{total_rainfall:.1f} mm", min(total_rainfall / 50 * 100, 100)),
]

for col, (label, value, width) in zip([k1, k2, k3, k4], card_data):
    with col:
        st.markdown(
            f"""
            <div class="fi-card">
                <div class="fi-label">{label}</div>
                <div class="fi-value">{value}</div>
                <div class="fi-line"><div style="width:{width:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    """
    <div class="fi-status">
        <span>U-NET <b>● COMPLETE</b></span>
        <span>PINN <b>● COMPLETE</b></span>
        <span>DEM <b>● READY</b></span>
        <span>RAINFALL <b>● READY</b></span>
    </div>
    """,
    unsafe_allow_html=True
)

# Main visual panel
st.markdown('<div class="fi-section">SATELLITE INTELLIGENCE</div>', unsafe_allow_html=True)
m1, m2 = st.columns([1.5, 1])

with m1:
    st.markdown('<div class="fi-panel">', unsafe_allow_html=True)
    st.markdown("**SENTINEL-1 FLOOD OVERVIEW**")
    fig, ax = plt.subplots(figsize=(9, 5))
    sar_display = sar[0].copy()
    sar_display[~np.isfinite(sar_display)] = np.nan
    ax.imshow(sar_display, cmap="gray")
    ax.imshow(
        np.ma.masked_where(probability < 0.5, probability),
        cmap="Blues",
        alpha=.62,
        vmin=.5,
        vmax=1
    )
    ax.axis("off")
    ax.set_title("SAR + U-Net Flood Overlay", color="white", fontsize=13)
    fig.patch.set_facecolor("#091a2a")
    ax.set_facecolor("#091a2a")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with m2:
    st.markdown('<div class="fi-panel">', unsafe_allow_html=True)
    st.markdown("**OVERALL ESTIMATED RISK**")
    st.markdown(
        f"""
        <div class="fi-risk">
            <div class="fi-risk-label">PROJECT-DEFINED DEPTH THRESHOLDS</div>
            <div class="fi-risk-value">{risk_level}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    risk_counts = [
        int((risk == 1).sum()),
        int((risk == 2).sum()),
        int((risk == 3).sum()),
        int((risk == 4).sum())
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.pie(
        risk_counts,
        labels=["Low", "Moderate", "High", "Severe"],
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
        wedgeprops={"width": .38}
    )
    ax.set_title("Risk Distribution", color="white")
    fig.patch.set_facecolor("#091a2a")
    ax.set_facecolor("#091a2a")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Maps
st.markdown('<div class="fi-section">FLOOD INTELLIGENCE MAPS</div>', unsafe_allow_html=True)
p1, p2 = st.columns(2)

with p1:
    st.markdown('<div class="fi-panel">', unsafe_allow_html=True)
    st.markdown("**CONTINUOUS FLOOD DEPTH**")
    fig, ax = plt.subplots(figsize=(7, 5))
    depth_display = np.ma.masked_where(depth <= 0, depth)
    im = ax.imshow(depth_display, cmap="turbo", vmin=0, vmax=max(maximum_depth, .4))
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.035, label="Depth (m)")
    fig.patch.set_facecolor("#091a2a")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with p2:
    st.markdown('<div class="fi-panel">', unsafe_allow_html=True)
    st.markdown("**FLOOD RISK MAP**")
    fig, ax = plt.subplots(figsize=(7, 5))
    risk_display = np.ma.masked_where(risk == 0, risk)
    im = ax.imshow(risk_display, cmap="RdYlGn_r", vmin=1, vmax=4)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=.035)
    fig.patch.set_facecolor("#091a2a")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Risk distribution numbers
st.markdown('<div class="fi-section">RISK DISTRIBUTION</div>', unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
for col, label, value in [
    (r1, "LOW", risk_counts[0]),
    (r2, "MODERATE", risk_counts[1]),
    (r3, "HIGH", risk_counts[2]),
    (r4, "SEVERE", risk_counts[3]),
]:
    with col:
        st.markdown(
            f"""
            <div class="fi-card">
                <div class="fi-label">{label}</div>
                <div class="fi-value">{value:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Rainfall
st.markdown('<div class="fi-section">24-HOUR RAINFALL PROFILE</div>', unsafe_allow_html=True)
st.markdown('<div class="fi-panel">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.plot(range(1, len(rainfall) + 1), rainfall, marker="o")
ax.set_xlabel("Hour")
ax.set_ylabel("Rainfall (mm)")
ax.grid(alpha=.2)
fig.patch.set_facecolor("#091a2a")
ax.set_facecolor("#091a2a")
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Pipeline
st.markdown('<div class="fi-section">AI PROCESSING PIPELINE</div>', unsafe_allow_html=True)
pipe_cols = st.columns(7)
steps = [
    ("🛰️", "Sentinel-1", "SAR"),
    ("⚙️", "Preprocess", "VV + VH"),
    ("🧠", "U-Net", "Segmentation"),
    ("🌊", "Probability", "Flood"),
    ("⛰️", "DEM + Rain", "Conditioning"),
    ("⚛️", "PINN", "Physics"),
    ("💧", "Depth + Risk", "Output"),
]
for col, (icon, title, sub) in zip(pipe_cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="fi-step">
                <div class="fi-step-icon">{icon}</div>
                <div class="fi-step-title">{title}</div>
                <div class="fi-step-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Downloads
st.markdown('<div class="fi-section">EXPORT ANALYSIS</div>', unsafe_allow_html=True)

# PDF REPORT
st.markdown('<div class="fi-section">FLOOD ANALYSIS REPORT</div>', unsafe_allow_html=True)

try:
    pdf_bytes = create_flood_report(
        sar_path=SAR_PATH,
        probability_path=PROB_PATH,
        depth_path=DEPTH_PATH,
        risk_path=RISK_PATH,
        rainfall_path=RAINFALL_PATH,
        rainfall_mm=float(total_rainfall),
    )

    st.download_button(
        "📄 Download Flood Analysis Report",
        data=pdf_bytes,
        file_name="Flood_Intelligence_Analysis_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

except Exception as e:
    st.error(f"Unable to generate PDF report: {e}")


d1, d2, d3 = st.columns(3)

with open(DEPTH_PATH, "rb") as f:
    depth_bytes = f.read()
with open(RISK_PATH, "rb") as f:
    risk_bytes = f.read()
with open(PROB_PATH, "rb") as f:
    probability_bytes = f.read()

d1.download_button("🌊 Flood Depth GeoTIFF", depth_bytes, "flood_depth_final.tif", "image/tiff")
d2.download_button("⚠️ Flood Risk GeoTIFF", risk_bytes, "flood_risk_map.tif", "image/tiff")
d3.download_button("🧠 Flood Probability GeoTIFF", probability_bytes, "flood_probability.tif", "image/tiff")

st.markdown(
    """
    <div class="fi-footer">
        🌊 FLOOD INTELLIGENCE | GEO-SPATIAL AI | PHYSICS-INFORMED FLOOD ANALYTICS
        <span style="float:right;">Academic Research Prototype</span>
    </div>
    """,
    unsafe_allow_html=True
)
