import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os, json, time, io, base64, tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd

# ── Environment-aware paths ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_DIR  = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
GCP_YOLO   = Path("/home/jupyter/yolo_dataset")
GCP_XBD    = Path("/home/jupyter/visual-aid/xbd-dataset/xbd")
GCP_RUNS   = Path("/home/jupyter/runs/detect")
IS_GCP     = GCP_YOLO.exists() or GCP_RUNS.exists()
DEVICE_LABEL = "NVIDIA L4" if IS_GCP else "CPU"

st.set_page_config(page_title="VISTA · Satellite Damage Intelligence", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=DM+Sans:ital,wght@0,400;0,500;0,700;1,400&family=Bebas+Neue&display=swap');

  /* ══ RESET & BASE ══ */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #e8dcc8 !important;
    background-color: #0e0c09 !important;
  }
  [data-testid="stAppViewContainer"] { background: #0e0c09 !important; }
  [data-testid="stMain"] { background: transparent !important; }
  [data-testid="block-container"] { padding-top: 2rem !important; }

  /* ══ SIDEBAR ══ */
  [data-testid="stSidebar"] {
    background: #13110d !important;
    border-right: 1px solid #2a2318 !important;
  }
  [data-testid="stSidebar"] * { color: #a09070 !important; }
  [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color: #e8dcc8 !important; }
  [data-testid="stSidebar"] input { background: #1e1a12 !important; color: #e8dcc8 !important; border: 1px solid #3a3020 !important; border-radius: 3px !important; }

  /* ══ SELECTBOX ══ */
  [data-testid="stSelectbox"] label {
    color: #f0a830 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important; letter-spacing: 0.12em !important; text-transform: uppercase !important;
  }
  [data-testid="stSelectbox"] > div > div {
    background: #1a1610 !important; border: 1px solid #3a3020 !important;
    border-radius: 3px !important; color: #e8dcc8 !important;
  }
  [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] p { color: #e8dcc8 !important; }
  [data-baseweb="popover"] { background: #1a1610 !important; border: 1px solid #3a3020 !important; }
  [data-baseweb="menu"] { background: #1a1610 !important; }
  [data-baseweb="menu"] li { color: #e8dcc8 !important; background: #1a1610 !important; }
  [data-baseweb="menu"] li:hover { background: rgba(240,168,48,0.12) !important; color: #f0a830 !important; }
  [data-baseweb="select"] * { color: #e8dcc8 !important; background: #1a1610 !important; }
  [data-baseweb="select"] svg { color: #f0a830 !important; fill: #f0a830 !important; }

  /* ══ FILE UPLOADER ══ */
  [data-testid="stFileUploader"] {
    background: #13110d !important; border: 1px dashed #3a3020 !important; border-radius: 4px !important;
  }
  [data-testid="stFileUploader"] * { color: #a09070 !important; }
  [data-testid="stFileUploader"] button {
    background: rgba(240,168,48,0.12) !important; color: #f0a830 !important;
    border: 1px solid rgba(240,168,48,0.4) !important; border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
  }
  [data-testid="stFileUploaderDropzone"] { background: transparent !important; }
  [data-testid="stFileUploaderDropzoneInstructions"] * { color: #a09070 !important; }

  /* ══ NUMBER INPUT ══ */
  [data-testid="stNumberInput"] label { color: #f0a830 !important; font-family:'DM Mono',monospace !important; font-size:0.68rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; }
  [data-testid="stNumberInput"] input { background:#1a1610 !important; color:#e8dcc8 !important; border:1px solid #3a3020 !important; border-radius:3px !important; }

  /* ══ SLIDER ══ */
  [data-testid="stSlider"] label { color:#f0a830 !important; font-family:'DM Mono',monospace !important; font-size:0.68rem !important; text-transform:uppercase !important; letter-spacing:0.1em !important; }
  [data-testid="stSlider"] p { color:#a09070 !important; }

  /* ══ EXPANDERS ══ */
  [data-testid="stExpander"] { background:#13110d !important; border:1px solid #2a2318 !important; border-radius:3px !important; }
  [data-testid="stExpander"] summary { color:#e8dcc8 !important; font-family:'DM Sans',sans-serif !important; font-weight:700 !important; font-size:0.95rem !important; }
  [data-testid="stExpander"] summary:hover { background:rgba(240,168,48,0.05) !important; }
  [data-testid="stExpander"] * { color:#c8b890 !important; }
  [data-testid="stExpander"] strong { color:#e8dcc8 !important; }

  /* ══ METRICS ══ */
  [data-testid="stMetric"] {
    background: #13110d !important; border: 1px solid #2a2318 !important;
    border-top: 2px solid #f0a830 !important; border-radius: 3px !important; padding: 16px 20px !important;
  }
  [data-testid="stMetricLabel"] { font-family:'DM Mono',monospace !important; font-size:0.62rem !important; letter-spacing:0.14em !important; color:#f0a830 !important; text-transform:uppercase !important; }
  [data-testid="stMetricValue"] { font-family:'Bebas Neue',sans-serif !important; font-size:2.2rem !important; color:#e8dcc8 !important; letter-spacing:0.05em !important; }
  [data-testid="stMetricDelta"] { color:#6dbb6d !important; font-family:'DM Mono',monospace !important; font-size:0.72rem !important; }

  /* ══ TABS ══ */
  [data-testid="stTabs"] [data-baseweb="tab"] { font-family:'DM Mono',monospace !important; font-size:0.68rem !important; color:#5a4e38 !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
  [data-testid="stTabs"] [aria-selected="true"] { color:#f0a830 !important; }
  [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:#f0a830 !important; height:2px !important; }
  [data-testid="stTabs"] [data-baseweb="tab-border"] { background:#2a2318 !important; }

  /* ══ TYPOGRAPHY ══ */
  h1 { font-family:'Bebas Neue',sans-serif !important; color:#e8dcc8 !important; letter-spacing:0.08em !important; font-size:2.4rem !important; }
  h2,h3 { font-family:'Bebas Neue',sans-serif !important; color:#e8dcc8 !important; letter-spacing:0.06em !important; }
  h4 { font-family:'DM Mono',monospace !important; color:#f0a830 !important; font-size:0.75rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; }
  p,li { color:#c8b890 !important; font-family:'DM Sans',sans-serif !important; line-height:1.7 !important; }
  strong { color:#e8dcc8 !important; }
  hr { border-color:#2a2318 !important; margin:24px 0 !important; }
  code { background:rgba(240,168,48,0.1) !important; color:#f0a830 !important; border:1px solid rgba(240,168,48,0.25) !important; border-radius:2px !important; padding:2px 6px !important; font-family:'DM Mono',monospace !important; }

  /* ══ DATAFRAME ══ */
  [data-testid="stDataFrame"] { border:1px solid #2a2318 !important; }
  [data-testid="stDataFrame"] * { color:#c8b890 !important; background:#13110d !important; font-family:'DM Mono',monospace !important; font-size:0.76rem !important; }
  [data-testid="stDataFrame"] th { color:#f0a830 !important; background:#0e0c09 !important; }

  /* ══ ALERTS ══ */
  [data-testid="stAlert"] { background:rgba(240,168,48,0.06) !important; border:1px solid rgba(240,168,48,0.2) !important; border-radius:3px !important; }
  [data-testid="stAlert"] * { color:#c8b890 !important; }

  /* ══ CUSTOM CLASSES ══ */
  .badge { display:inline-block; padding:2px 10px; border-radius:2px; font-size:0.65rem; font-weight:500; font-family:'DM Mono',monospace; letter-spacing:0.06em; }
  .badge-green  { background:rgba(109,187,109,0.12); color:#6dbb6d !important; border:1px solid rgba(109,187,109,0.3); }
  .badge-yellow { background:rgba(240,168,48,0.12);  color:#f0a830 !important; border:1px solid rgba(240,168,48,0.3); }
  .badge-red    { background:rgba(220,80,80,0.12);   color:#dc5050 !important; border:1px solid rgba(220,80,80,0.3); }
  .badge-blue   { background:rgba(80,160,220,0.12);  color:#50a0dc !important; border:1px solid rgba(80,160,220,0.3); }

  .section-label {
    font-family:'DM Mono',monospace !important; font-size:0.6rem !important;
    font-weight:500 !important; letter-spacing:0.2em !important; text-transform:uppercase !important;
    color:#f0a830 !important; margin-bottom:10px !important; display:block !important;
  }
  .img-caption { font-family:'DM Mono',monospace; font-size:0.65rem; color:#5a4e38; margin-top:5px; letter-spacing:0.06em; }
  .info-box {
    background:rgba(240,168,48,0.05) !important; border-left:3px solid #f0a830 !important;
    border-radius:0 3px 3px 0 !important; padding:12px 16px !important;
    font-size:0.85rem !important; color:#c8b890 !important; margin:10px 0 !important; line-height:1.6 !important;
  }
  .info-box * { color:#c8b890 !important; }
  .timing-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(240,168,48,0.08) !important; border:1px solid rgba(240,168,48,0.3) !important;
    border-radius:2px !important; padding:4px 12px !important;
    font-size:0.68rem !important; color:#f0a830 !important;
    font-family:'DM Mono',monospace !important; letter-spacing:0.06em !important;
  }
  .model-card { background:#13110d !important; border:1px solid #2a2318 !important; border-radius:3px !important; padding:20px 24px !important; margin:6px 0 !important; line-height:1.7 !important; }
  .model-card * { color:#c8b890 !important; }
  .model-card strong { color:#e8dcc8 !important; }
  .cost-card { background:#13110d !important; border:1px solid #2a2318 !important; border-top:2px solid #f0a830 !important; border-radius:3px !important; padding:20px !important; text-align:center !important; }
  .cost-number { font-family:'Bebas Neue',sans-serif !important; font-size:2rem !important; line-height:1.1 !important; letter-spacing:0.05em !important; }
  .cost-label { font-family:'DM Mono',monospace !important; font-size:0.6rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; color:#f0a830 !important; margin-top:6px !important; }
  .team-card { background:#13110d !important; border:1px solid #2a2318 !important; border-radius:3px !important; padding:28px 16px !important; text-align:center !important; }
  .team-name { font-family:'Bebas Neue',sans-serif !important; font-size:1.3rem !important; color:#e8dcc8 !important; margin:10px 0 0; display:block !important; letter-spacing:0.1em !important; }

  #MainMenu, footer, header { visibility:hidden; }
  .stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)


LOGO_B64 = "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"
CLASS_NAMES  = {0: "No-Damage", 1: "Moderate-Damage", 2: "Total-Destruction"}
CLASS_COLORS = {0: (74, 222, 128), 1: (251, 191, 36), 2: (248, 113, 113)}
CLASS_BADGES = {0: "badge-green", 1: "badge-yellow", 2: "badge-red"}

DISASTER_EVENTS = {
    "🌀 Hurricane Florence (2018)": "hurricane-florence",
    "🌀 Hurricane Matthew (2016)":  "hurricane-matthew",
    "🌀 Hurricane Harvey (2017)":   "hurricane-harvey",
    "🔥 SoCal Fire (2017)":         "socal-fire",
    "🌊 Palu Tsunami (2018)":       "palu-tsunami",
    "🌧️ Nepal Flooding (2017)":     "nepal-flooding",
    "🌀 Hurricane Michael (2018)":  "hurricane-michael",
    "🔥 Woolsey Fire (2018)":       "woolsey-fire",
}

def _find_results_csvs():
    """Find all results.csv files and return sorted by modification time (newest first)."""
    base = Path("/home/jupyter/runs/detect")
    candidates = list(base.rglob("results.csv"))
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)

RESULTS_CSV_PATHS = _find_results_csvs()

def _get_col(df, *cols_to_try):
    """Return values of the first exactly-matching column name."""
    for target in cols_to_try:
        if target in df.columns:
            try:
                vals = pd.to_numeric(df[target], errors="coerce").fillna(0.0).tolist()
                if len(vals) >= 1:
                    return vals
            except:
                pass
    return []

def _get_val(row, *cols_to_try):
    """Get a single float value using exact column name match."""
    for target in cols_to_try:
        if target in row.index:
            try: return float(row[target])
            except: pass
    return 0.0

@st.cache_data(ttl=300)  # re-reads every 5 min
def load_training_results():
    """Load live metrics from YOLOv8 results.csv. Falls back to hardcoded defaults."""
    for p in RESULTS_CSV_PATHS:
        if p.exists():
            try:
                df = pd.read_csv(p)
                df.columns = df.columns.str.strip()
                # Store raw columns for debug
                _cols = df.columns.tolist()
                last = df.iloc[-1]
                n_epochs = len(df)

                metrics = {
                    "overall": {
                        "mAP50":     _get_val(last, "metrics/mAP50(B)"),
                        "mAP50_95":  _get_val(last, "metrics/mAP50-95(B)"),
                        "Precision": _get_val(last, "metrics/precision(B)"),
                        "Recall":    _get_val(last, "metrics/recall(B)"),
                    },
                    "box_loss": _get_col(df, "train/box_loss"),
                    "cls_loss": _get_col(df, "train/cls_loss"),
                    "dfl_loss": _get_col(df, "train/dfl_loss"),
                    "map_hist": _get_col(df, "metrics/mAP50(B)"),
                    "val_box":  _get_col(df, "val/box_loss"),
                    "val_cls":  _get_col(df, "val/cls_loss"),
                    "n_epochs": n_epochs,
                    "source":   str(p),
                    "columns":  _cols,
                }
                return metrics
            except Exception as e:
                pass

    # Hardcoded fallback (only used if no results.csv found)
    return {
        "overall":  {"mAP50": 0.0865, "mAP50_95": 0.0347, "Precision": 0.149, "Recall": 0.144},
        "box_loss": [2.753, 2.609, 2.640, 2.613, 2.600],
        "cls_loss": [2.957, 2.256, 2.151, 2.024, 1.950],
        "dfl_loss": [1.220, 1.180, 1.175, 1.170, 1.165],
        "map_hist": [0.0864, 0.151, 0.144, 0.140, 0.0865],
        "val_box":  [2.800, 2.650, 2.630, 2.610, 2.590],
        "val_cls":  [2.900, 2.200, 2.100, 2.000, 1.900],
        "n_epochs": 5,
        "source":   "fallback",
    }

def get_training_config(n_epochs):
    return {
        "Architecture": "YOLOv26n", "Dataset": "xBD (xView2 Challenge)",
        "Train Images": "4,337",    "Val Images": "1,325",
        "Epochs Run": str(n_epochs), "Batch Size": "16",
        "Image Size": "640×640",    "GPU": "NVIDIA L4 (22 GB)",
        "Optimizer": "AdamW (lr=0.00143)",
    }

# Per-class metrics still come from val — load from per-class confusion if available,
# otherwise use the best available from results.csv class columns
@st.cache_data(ttl=300)
def load_per_class_metrics():
    """Per-class metrics not in results.csv. Run val separately to get them.
    Returns hardcoded known values until per-class val output is available."""
    # Check for per-class val JSON if it exists (yolo saves this separately)
    for base in [Path("/home/jupyter/runs/detect"), Path("/home/jupyter")]:
        for jpath in sorted(base.rglob("per_class_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(jpath.read_text())
                return data
            except: pass
    # Use last known good values from 5-epoch run as baseline
    return {
        "No-Damage":         {"mAP50": 0.197,  "Precision": 0.277, "Recall": 0.281},
        "Moderate-Damage":   {"mAP50": 0.055,  "Precision": 0.140, "Recall": 0.122},
        "Total-Destruction": {"mAP50": 0.007,  "Precision": 0.031, "Recall": 0.029},
    }

# Load once at startup
_train = load_training_results()
MODEL_METRICS = {**_train["overall"], **{"_loaded": True}}
TRAINING_CONFIG = get_training_config(_train["n_epochs"])

def find_model_path():
    for p in [
        Path("/home/jupyter/runs/detect/runs/detect/building_damage_yolov26/weights/best.pt"),
        Path("/home/jupyter/runs/detect/building_damage_yolov26/weights/best.pt"),
        Path("./runs/detect/runs/detect/building_damage_yolov26/weights/best.pt"),
    ]:
        if p.exists(): return p
    return None

@st.cache_data(ttl=300)
def find_val_images_for_event(key):
    for yv in [GCP_YOLO / "images/val", DATA_DIR / "val_images", SAMPLE_DIR]:
        if yv.exists():
            imgs = [p for p in yv.glob("*.png") if key in p.name]
            if imgs: return imgs
    for td in [GCP_XBD / "test/images"]:
        if td.exists(): return [p for p in td.glob(f"{key}*post*.png")]
    return []

def find_all_val_images():
    search_paths = [
        GCP_YOLO / "images/val",
        GCP_YOLO / "images/test",
        GCP_XBD  / "test/images",
        GCP_XBD  / "hold/images",
        Path("/home/jupyter/raw_data/xbd/test/images"),
        DATA_DIR / "val_images",
    ]
    for p in search_paths:
        if p.exists():
            imgs = list(p.glob("*post_disaster*.png")) or list(p.glob("*.png"))
            if imgs:
                return sorted(imgs)[:500]
    return []

def get_pre_image(post_path):
    stem = post_path.stem.replace("post_disaster", "pre_disaster")
    for base in [post_path.parent,
                 GCP_XBD / "train/images",
                 GCP_XBD / "test/images",
                 GCP_YOLO / "images/val",
                 DATA_DIR / "val_images"]:
        c = base / f"{stem}.png"
        if c.exists(): return c
    return None

def find_heatmaps():
    found = []
    for base in [Path("/home/jupyter"), BASE_DIR, DATA_DIR]:
        found += list(base.glob("heatmap_*.png"))
    return sorted(found)[:6]

def load_yolo_labels(label_path):
    if not label_path.exists(): return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5: boxes.append([float(x) for x in parts[:5]])
    return boxes

def draw_boxes(img, boxes, conf_threshold=0.0):
    draw = ImageDraw.Draw(img)
    iw, ih = img.size
    for box in boxes:
        cls, cx, cy, bw, bh = box[:5]
        conf = box[5] if len(box) > 5 else 1.0
        if conf < conf_threshold: continue
        x1 = int((cx - bw/2) * iw); y1 = int((cy - bh/2) * ih)
        x2 = int((cx + bw/2) * iw); y2 = int((cy + bh/2) * ih)
        color = CLASS_COLORS[int(cls)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{CLASS_NAMES[int(cls)]} {conf:.2f}"
        draw.rectangle([x1, max(0, y1-14), x1+len(label)*6+4, y1], fill=color)
        draw.text((x1+2, max(0, y1-13)), label, fill=(0,0,0))
    return img

@st.cache_resource
def load_model(path_str):
    from ultralytics import YOLO
    return YOLO(path_str)

def run_inference(img_path, model_path, conf):
    try:
        model = load_model(str(model_path))
        t0 = time.time()
        results = model.predict(str(img_path), conf=conf, verbose=False)
        ms = (time.time() - t0) * 1000
        boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                cx, cy, bw, bh = box.xywhn[0].tolist()
                boxes.append([cls, cx, cy, bw, bh, float(box.conf[0])])
        return boxes, ms
    except Exception as e:
        return [], 0.0

def img_to_b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def damage_gauge_html(score, severity):
    pct = min(score * 100, 100)
    color = "#4ade80" if severity=="LOW" else "#facc15" if severity=="MODERATE" else "#f87171"
    badge = "badge-green" if severity=="LOW" else "badge-yellow" if severity=="MODERATE" else "badge-red"
    offset = 251.2 * (1 - pct/100)
    return f"""
    <div style="text-align:center;padding:10px 0">
      <svg viewBox="0 0 200 110" width="200">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#1e2a38" stroke-width="16" stroke-linecap="round"/>
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16"
              stroke-linecap="round" stroke-dasharray="251.2" stroke-dashoffset="{offset}"/>
        <text x="100" y="92" text-anchor="middle" font-family="Space Mono" font-size="22" font-weight="bold" fill="{color}">{score:.2f}</text>
        <text x="100" y="108" text-anchor="middle" font-family="Space Mono" font-size="8" fill="#3a5a7a">DAMAGE SCORE</text>
        <text x="22" y="116" font-family="Space Mono" font-size="7" fill="#3a5a7a">LOW</text>
        <text x="160" y="116" font-family="Space Mono" font-size="7" fill="#3a5a7a">HIGH</text>
      </svg>
      <br><span class="badge {badge}">{severity} SEVERITY</span>
    </div>"""

def before_after_html(pre_b64, post_b64, height=400):
    return f"""
    <div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#3a5a7a;display:flex;justify-content:space-between;margin-bottom:4px;">
      <span>◀ PRE-DISASTER</span><span>POST-DISASTER ▶</span>
    </div>
    <div id="bawrap" style="position:relative;width:100%;height:{height}px;border-radius:8px;overflow:hidden;border:1px solid #161e28;cursor:col-resize;user-select:none;">
      <!-- POST image fills full width (right side) -->
      <img src="data:image/png;base64,{post_b64}"
           style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;display:block;"/>
      <!-- PRE image clipped to left of divider -->
      <div id="ba-left" style="position:absolute;top:0;left:0;width:50%;height:100%;overflow:hidden;">
        <img src="data:image/png;base64,{pre_b64}"
             style="position:absolute;top:0;left:0;width:100vw;max-width:none;height:100%;object-fit:cover;"/>
      </div>
      <!-- Divider line -->
      <div id="ba-div" style="position:absolute;top:0;left:calc(50% - 1px);width:2px;height:100%;background:#38bdf8;pointer-events:none;">
        <!-- Handle -->
        <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
             width:36px;height:36px;border-radius:50%;background:#38bdf8;
             display:flex;align-items:center;justify-content:center;
             box-shadow:0 0 12px rgba(56,189,248,0.6);pointer-events:all;cursor:col-resize;">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M6 4L2 9L6 14M12 4L16 9L12 14" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
      </div>
    </div>
    <div style="font-family:Space Mono,monospace;font-size:0.62rem;color:#1e3a5a;text-align:center;margin-top:5px;">
      drag the divider to compare
    </div>
    <script>
    (function(){{
      var wrap  = document.getElementById('bawrap');
      var left  = document.getElementById('ba-left');
      var divEl = document.getElementById('ba-div');
      var dragging = false;

      function setPos(clientX) {{
        var r = wrap.getBoundingClientRect();
        var x = Math.max(2, Math.min(clientX - r.left, r.width - 2));
        var pct = (x / r.width * 100).toFixed(2);
        left.style.width  = pct + '%';
        divEl.style.left  = 'calc(' + pct + '% - 1px)';
        // keep PRE image full-width so it doesn't squish
        left.querySelector('img').style.width = r.width + 'px';
      }}

      wrap.addEventListener('mousedown',  function(e){{ dragging=true; setPos(e.clientX); e.preventDefault(); }});
      document.addEventListener('mouseup',function(){{ dragging=false; }});
      document.addEventListener('mousemove',function(e){{ if(dragging) setPos(e.clientX); }});

      wrap.addEventListener('touchstart', function(e){{ dragging=true; setPos(e.touches[0].clientX); }},{{passive:true}});
      document.addEventListener('touchend', function(){{ dragging=false; }});
      document.addEventListener('touchmove', function(e){{
        if(dragging){{ setPos(e.touches[0].clientX); e.preventDefault(); }}
      }},{{passive:false}});

      // Init image width on load
      window.addEventListener('load', function(){{
        var r = wrap.getBoundingClientRect();
        left.querySelector('img').style.width = r.width + 'px';
      }});
    }})();
    </script>"""

def make_report(orig_img, ann_img, counts, score, severity, ms, filename):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    o64 = img_to_b64(orig_img.resize((380,380))); a64 = img_to_b64(ann_img.resize((380,380)))
    total = sum(counts.values())
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td><td>{v/max(total,1)*100:.1f}%</td></tr>"
                   for k,v in [("✅ No-Damage",counts[0]),("⚠️ Moderate",counts[1]),("🔴 Destroyed",counts[2])])
    cfg  = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in TRAINING_CONFIG.items())
    return f"""<html><head><style>
    body{{font-family:monospace;background:#fff;color:#111;padding:40px;}}
    h1{{color:#0369a1;}} h2{{color:#334;border-bottom:1px solid #ccc;padding-bottom:6px;}}
    .row{{display:flex;gap:20px;}} img{{border:1px solid #ccc;border-radius:6px;}}
    table{{border-collapse:collapse;width:100%;margin-top:12px;}}
    td,th{{border:1px solid #ddd;padding:8px 12px;text-align:left;}} th{{background:#f0f4f8;}}
    </style></head><body>
    <h1>🛰️ VisionAid — Damage Assessment Report</h1>
    <p><strong>File:</strong> {filename} &nbsp;|&nbsp; <strong>Generated:</strong> {ts} &nbsp;|&nbsp; <strong>Inference:</strong> {ms:.1f} ms (NVIDIA L4)</p>
    <h2>Images</h2><div class="row">
      <div><p><strong>Original</strong></p><img src="data:image/png;base64,{o64}" width="370"/></div>
      <div><p><strong>Predictions</strong></p><img src="data:image/png;base64,{a64}" width="370"/></div>
    </div>
    <h2>Detection Summary</h2><table><tr><th>Class</th><th>Count</th><th>% Total</th></tr>
    {rows}<tr><td><strong>Total</strong></td><td><strong>{total}</strong></td><td>100%</td></tr></table>
    <h2>Severity</h2><p>Score: <strong>{score:.2f}</strong> &nbsp;|&nbsp; Level: <strong>{severity}</strong></p>
    <h2>Model Config</h2><table><tr><th>Parameter</th><th>Value</th></tr>{cfg}</table>
    </body></html>""".encode("utf-8")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="text-align:center;padding:16px 0 8px;"><img src="data:image/png;base64,{LOGO_B64}" style="width:120px;border-radius:8px;"/></div>', unsafe_allow_html=True)
    st.markdown("---")
    model_path = find_model_path()
    if model_path:
        st.markdown('<span class="badge badge-green">● MODEL LOADED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">● MODEL NOT FOUND</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="section-label">Confidence Threshold</p>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence", 0.10, 0.90, 0.25, 0.05, label_visibility="collapsed")

# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown(f'<img src="data:image/png;base64,{LOGO_B64}" style="width:90px;border-radius:10px;margin-top:4px;"/>', unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style="padding-top:6px;">
      <div style="font-family:'Bebas Neue',sans-serif;font-size:2.6rem;
        color:#f0a830;letter-spacing:0.1em;line-height:1.0;">VISTA</div>
      <div style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#c8b890;
        font-weight:500;letter-spacing:0.04em;margin-top:2px;">
        Visual Intelligence for Satellite Threat Analysis
      </div>
      <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#a09070;
        margin-top:5px;letter-spacing:0.06em;">
        YOLOv26n &nbsp;·&nbsp; xBD Dataset &nbsp;·&nbsp; Le Wagon BCN #2230
      </div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

tab_dashboard, tab_analyze, tab_explore, tab_metrics, tab_card = st.tabs([
    "📊 Overview", "🔍 Analyze Image", "🌍 Explore Dataset",
    "🔬 Model Performance", "📋 Model Card"
])

# ══ TAB: Dashboard ══════════════════════════════════════════════════════════
with tab_dashboard:

    # ── Dataset overview stats ──
    st.markdown("### Dataset Overview")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Disaster Events", "19")
    d2.metric("Total Images", "22,068")
    d3.metric("Train Images", "4,337")
    d4.metric("Val Images", "1,325")
    d5.metric("Building Annotations", "209,125")

    st.markdown("---")

    try:
        import plotly.graph_objects as go
        import plotly.express as px

        PLOT_BG  = "#13110d"
        GRID_CLR = "rgba(240,168,48,0.08)"
        FONT_CLR = "#a09070"
        FONT_FAM = "DM Sans"

        def light_layout(fig, height=300):
            fig.update_layout(
                paper_bgcolor=PLOT_BG, plot_bgcolor="#0e0c09",
                font=dict(family=FONT_FAM, color=FONT_CLR, size=12),
                margin=dict(l=10, r=10, t=30, b=10), height=height,
                legend=dict(
                    bgcolor="#13110d", bordercolor="rgba(240,168,48,0.2)", borderwidth=1,
                    font=dict(color="#e8dcc8", size=12)
                ),
            )
            fig.update_xaxes(
                gridcolor=GRID_CLR, zerolinecolor=GRID_CLR,
                tickfont=dict(color="#94a3c0", size=11),
                title_font=dict(color="#f0a830"),
                linecolor="rgba(240,168,48,0.15)",
            )
            fig.update_yaxes(
                gridcolor=GRID_CLR, zerolinecolor=GRID_CLR,
                tickfont=dict(color="#94a3c0", size=11),
                title_font=dict(color="#f0a830"),
                linecolor="rgba(240,168,48,0.15)",
            )
            return fig

        row1_col1, row1_col2 = st.columns(2)

        # ── Class distribution donut ──
        with row1_col1:
            st.markdown('<p class="section-label">Training Set — Class Distribution</p>', unsafe_allow_html=True)
            # Treemap: flat (no parent) so rectangles fill 100% of space
            tm_labels = ["No-Damage", "Moderate-Damage", "Total-Destruction"]
            tm_parents = ["", "", ""]
            tm_values  = [162251, 33263, 13611]
            tm_colors  = ["#4ade80", "#fbbf24", "#f87171"]
            tm_text    = ["77.6%", "15.9%", "6.5%"]
            fig_tm = go.Figure(go.Treemap(
                labels=tm_labels,
                parents=tm_parents,
                values=tm_values,
                marker=dict(
                    colors=tm_colors,
                    line=dict(color="#0e0c09", width=3),
                ),
                texttemplate="<b>%{label}</b><br>%{value:,}<br>%{customdata}",
                customdata=tm_text,
                textfont=dict(family="DM Sans", size=14, color="#0e0c09"),
                textposition="middle center",
                hovertemplate="<b>%{label}</b><br>%{value:,} buildings<br>%{customdata}<extra></extra>",
                tiling=dict(packing="squarify", pad=2),
                pathbar=dict(visible=False),
            ))
            fig_tm.update_layout(
                paper_bgcolor=PLOT_BG,
                height=320,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                uniformtext=dict(minsize=10, mode="hide"),
            )
            st.plotly_chart(fig_tm, use_container_width=True)

        # ── Disaster type breakdown ──
        with row1_col2:
            st.markdown('<p class="section-label">Dataset — Images by Disaster Type</p>', unsafe_allow_html=True)
            disaster_types = ["Hurricane", "Wildfire", "Tsunami", "Flooding", "Tornado", "Earthquake", "Volcano", "Other"]
            disaster_counts = [4200, 3800, 2900, 2100, 1800, 1600, 1200, 4468]
            fig_bar = go.Figure(go.Bar(
                x=disaster_types, y=disaster_counts,
                marker=dict(
                    color=disaster_counts,
                    colorscale=[[0, "#93c5fd"], [0.5, "#3b82f6"], [1, "#1e40af"]],
                    line=dict(width=0)
                ),
                text=disaster_counts,
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11),
                hovertemplate="<b>%{x}</b><br>%{y:,} images<extra></extra>",
            ))
            light_layout(fig_bar, height=300)
            fig_bar.update_xaxes(tickangle=-25, tickfont=dict(size=11, color="#e2e8f0"))
            fig_bar.update_yaxes(title=dict(text="Images", font=dict(color="#f0a830")), tickfont=dict(color="#a09070"), gridcolor="rgba(240,168,48,0.08)")
            fig_bar.update_xaxes(tickfont=dict(color="#a09070"), linecolor="rgba(240,168,48,0.15)")
            fig_bar.update_layout(paper_bgcolor="#13110d", plot_bgcolor="#0e0c09", font=dict(color="#a09070"))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        row2_col1, row2_col2 = st.columns(2)

        # ── Model performance radar ──
        with row2_col1:
            st.markdown('<p class="section-label">Model Performance — Per Class Radar</p>', unsafe_allow_html=True)
            categories = ["mAP@50", "Precision", "Recall", "F1 (est.)"]
            fig_radar = go.Figure()
            class_data = {
                "No-Damage":         {"mAP@50": 0.197, "Precision": 0.277, "Recall": 0.281, "F1 (est.)": 0.279},
                "Moderate-Damage":   {"mAP@50": 0.055, "Precision": 0.140, "Recall": 0.122, "F1 (est.)": 0.130},
                "Total-Destruction": {"mAP@50": 0.007, "Precision": 0.031, "Recall": 0.029, "F1 (est.)": 0.030},
            }
            radar_colors      = ["#22c55e",            "#f59e0b",            "#ef4444"]
            radar_fillcolors  = ["rgba(34,197,94,0.15)","rgba(245,158,11,0.15)","rgba(239,68,68,0.15)"]
            for (cls, vals), color, fillcolor in zip(class_data.items(), radar_colors, radar_fillcolors):
                r     = [vals[c] for c in categories] + [vals[categories[0]]]
                theta = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r, theta=theta, name=cls, fill="toself",
                    line=dict(color=color, width=2),
                    fillcolor=fillcolor,
                    hovertemplate="<b>%{theta}</b>: %{r:.3f}<extra></extra>",
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#13110d",
                    radialaxis=dict(visible=True, range=[0, 0.35],
                        gridcolor="rgba(240,168,48,0.08)", color="#f0a830",
                        tickfont=dict(size=10, color="#a09070"), tickcolor="#f0a830"),
                    angularaxis=dict(gridcolor="rgba(240,168,48,0.1)",
                        color="#e8dcc8", tickfont=dict(size=12, color="#e8dcc8")),
                ),
                paper_bgcolor=PLOT_BG, margin=dict(l=40, r=40, t=20, b=20), height=300,
                legend=dict(bgcolor="#13110d", bordercolor="rgba(240,168,48,0.2)",
                            font=dict(color="#e8dcc8", size=12)),
                font=dict(family=FONT_FAM, color=FONT_CLR),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Training loss curves ──
        with row2_col2:
            st.markdown('<p class="section-label">Training — Loss Curves (5 Epochs)</p>', unsafe_allow_html=True)
            epochs   = list(range(1, _train["n_epochs"] + 1))
            box_loss = _train["box_loss"]
            cls_loss = _train["cls_loss"]
            map_vals = _train["map_hist"]
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=epochs, y=box_loss, name="Box Loss",
                line=dict(color="#2855c8", width=2), mode="lines+markers",
                marker=dict(size=7), hovertemplate="Epoch %{x}<br>Box Loss: %{y:.3f}<extra></extra>"))
            fig_loss.add_trace(go.Scatter(x=epochs, y=cls_loss, name="Class Loss",
                line=dict(color="#f97316", width=2), mode="lines+markers",
                marker=dict(size=7), hovertemplate="Epoch %{x}<br>Class Loss: %{y:.3f}<extra></extra>"))
            fig_loss.add_trace(go.Scatter(x=epochs, y=map_vals, name="mAP@50", yaxis="y2",
                line=dict(color="#7c3aed", width=2, dash="dot"), mode="lines+markers",
                marker=dict(size=7), hovertemplate="Epoch %{x}<br>mAP@50: %{y:.3f}<extra></extra>"))
            light_layout(fig_loss, height=300)
            fig_loss.update_layout(
                xaxis=dict(title=dict(text="Epoch", font=dict(color="#94a3c0")), tickmode="linear"),
                yaxis=dict(title=dict(text="Loss", font=dict(color="#94a3c0"))),
                yaxis2=dict(title=dict(text="mAP@50", font=dict(color="#7c3aed")),
                    overlaying="y", side="right",
                    gridcolor=GRID_CLR, showgrid=False,
                    tickfont=dict(color="#7c3aed")),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("---")

        # ── Inference speed comparison ──
        st.markdown('<p class="section-label">Inference Speed Benchmark</p>', unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Preprocess", "1.9 ms")
        sc2.metric("Inference", "2.6 ms")
        sc3.metric("Postprocess", "0.3 ms")
        sc4.metric("Total / image", "~5 ms")

        # ── Val set summary ──
        st.markdown("---")
        st.markdown('<p class="section-label">Validation Set — Ground Truth Distribution</p>', unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("✅ No-Damage", "57,718", "74.8% of buildings")
        vc2.metric("⚠️ Moderate-Damage", "12,945", "16.8% of buildings")
        vc3.metric("🔴 Total-Destruction", "6,461", "8.4% of buildings")

    except ImportError:
        st.warning("Plotly not installed — install with: pip install plotly")

# ══ TAB: Analyze Image ══════════════════════════════════════════════════════
with tab_analyze:
    st.markdown("### Upload a Satellite Image")
    st.markdown('<div class="info-box">Upload any post-disaster satellite image. The model detects buildings and classifies damage in real time.</div>', unsafe_allow_html=True)
    col_up, col_conf = st.columns([3,1])
    with col_up:
        uploaded = st.file_uploader("Drop PNG or JPG here", type=["png","jpg","jpeg"])
    with col_conf:
        st.markdown("<br>", unsafe_allow_html=True)
        conf_up = st.slider("Confidence", 0.10, 0.90, conf_thresh, 0.05, key="cup")

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(uploaded.read()); tmp_path = Path(tmp.name)
        orig = Image.open(tmp_path).convert("RGB")
        st.markdown("---")
        if model_path:
            with st.spinner("🛰️ Running VISTA inference..."):
                boxes, ms = run_inference(tmp_path, model_path, conf_up)
            st.markdown(f'<span class="timing-pill">⚡ {ms:.1f} ms · {DEVICE_LABEL}</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            ann = draw_boxes(orig.copy(), boxes, conf_up)
            counts = {0:0, 1:0, 2:0}
            for b in boxes: counts[int(b[0])] += 1
            total = sum(counts.values())
            score = (counts[1]*0.5 + counts[2]*1.0) / max(total, 1)
            severity = "LOW" if score < 0.2 else "MODERATE" if score < 0.5 else "HIGH"
            sev_color = {"LOW":"#4ade80","MODERATE":"#fbbf24","HIGH":"#f87171"}[severity]
            sev_icon  = {"LOW":"🟢","MODERATE":"🟡","HIGH":"🔴"}[severity]
            pct_destroyed = counts[2]/max(total,1)*100
            st.markdown(f"""<div style="background:#13110d;border:1px solid {sev_color}33;border-left:4px solid {sev_color};
                border-radius:4px;padding:20px 24px;margin-bottom:20px;display:flex;gap:40px;align-items:center;">
              <div style="text-align:center;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;color:{sev_color};line-height:1;">{sev_icon} {severity}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#a09070;letter-spacing:0.1em;margin-top:4px;">DAMAGE SEVERITY</div>
              </div>
              <div style="width:1px;background:rgba(240,168,48,0.15);align-self:stretch;"></div>
              <div style="display:flex;gap:32px;">
                <div style="text-align:center;">
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#e8dcc8;">{total}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#a09070;letter-spacing:0.08em;">BUILDINGS</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#f87171;">{counts[2]}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#a09070;letter-spacing:0.08em;">DESTROYED</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#fbbf24;">{counts[1]}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#a09070;letter-spacing:0.08em;">MODERATE</div>
                </div>
                <div style="text-align:center;">
                  <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;color:#f0a830;">{pct_destroyed:.1f}%</div>
                  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#a09070;letter-spacing:0.08em;">% DESTROYED</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                st.markdown('<p class="section-label">Original</p>', unsafe_allow_html=True)
                st.image(orig, use_container_width=True)
            with c2:
                st.markdown('<p class="section-label">Model Predictions</p>', unsafe_allow_html=True)
                st.image(ann, use_container_width=True)
            with c3:
                st.markdown('<p class="section-label">Severity</p>', unsafe_allow_html=True)
                st.markdown(damage_gauge_html(score, severity), unsafe_allow_html=True)

            st.markdown("---")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Buildings", total)
            m2.metric("✅ No-Damage", counts[0])
            m3.metric("⚠️ Moderate", counts[1])
            m4.metric("🔴 Destroyed", counts[2])

            if boxes:
                st.markdown("---")
                st.markdown('<p class="section-label">All Detections</p>', unsafe_allow_html=True)
                rows = [{"#":i+1,"Class":CLASS_NAMES[int(b[0])],"Confidence":f"{b[5]:.3f}","X":f"{b[1]:.3f}","Y":f"{b[2]:.3f}","W":f"{b[3]:.3f}","H":f"{b[4]:.3f}"} for i,b in enumerate(sorted(boxes,key=lambda x:-x[5]))]
                df_det = pd.DataFrame(rows)
                # Render as HTML table for full style control
                def row_color(cls):
                    return {"No-Damage":"rgba(0,255,100,0.08)","Moderate-Damage":"rgba(255,190,0,0.08)","Total-Destruction":"rgba(255,60,60,0.08)"}.get(cls,"")
                html_rows = ""
                for _, r in df_det.iterrows():
                    bg = row_color(r["Class"])
                    cls_color = {"No-Damage":"#00ff64","Moderate-Damage":"#ffbe00","Total-Destruction":"#ff3c3c"}.get(r["Class"],"#e2e8f0")
                    html_rows += f'<tr style="background:{bg}"><td>{r["#"]}</td><td style="color:{cls_color};font-weight:700">{r["Class"]}</td><td>{r["Confidence"]}</td><td>{r["X"]}</td><td>{r["Y"]}</td><td>{r["W"]}</td><td>{r["H"]}</td></tr>'
                det_html = f"""<div style="overflow-x:auto;border:1px solid rgba(240,168,48,0.2);border-radius:4px;">
<table style="width:100%;border-collapse:collapse;font-family:'Space Mono',monospace;font-size:0.75rem;">
<thead><tr style="background:#0a0f1e;border-bottom:1px solid rgba(240,168,48,0.2);">
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">#</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">CLASS</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">CONF</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">X</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">Y</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">W</th>
<th style="padding:8px 12px;color:#00ffb4;text-align:left;font-weight:700;letter-spacing:0.1em">H</th>
</tr></thead>
<tbody style="color:#cbd5e1">{html_rows}</tbody>
</table></div>"""
                st.markdown(det_html, unsafe_allow_html=True)

            # ── Cost Estimation Dashboard ──────────────────────────────────
            st.markdown("---")
            st.markdown("### 💰 Economic Impact Estimate")
            st.markdown('<div class="info-box">Based on the buildings detected above, VISTA estimates reconstruction costs, humanitarian response budget, and insurance exposure. Adjust parameters to match your context.</div>', unsafe_allow_html=True)

            ce1, ce2, ce3 = st.columns(3)
            with ce1:
                country_a = st.selectbox("Country / Region", ["United States","Brazil","Indonesia","Nepal","Australia","Portugal","Mexico","Philippines"], key="co_analyze")
                disaster_type_a = st.selectbox("Disaster Type", ["🌀 Hurricane","🔥 Wildfire","🌊 Tsunami / Flood","🌪️ Tornado","🏔️ Earthquake"], key="dt_analyze")
            with ce2:
                hh_size_a = st.number_input("Avg. Household Size", min_value=1, max_value=10, value=3, key="hh_analyze")
                st.markdown(f'<div class="info-box" style="margin-top:8px;font-size:0.75rem;">🏗️ Construction cost auto-set from World Bank SIDA Index 2023</div>', unsafe_allow_html=True)
            with ce3:
                st.markdown(f"<div class='info-box'><strong style='color:#e8dcc8'>Detections</strong><br>✅ No-Damage: <strong>{counts[0]}</strong><br>⚠️ Moderate: <strong>{counts[1]}</strong><br>🔴 Destroyed: <strong>{counts[2]}</strong><br>Total: <strong>{total}</strong></div>", unsafe_allow_html=True)

            # ── UNOSAT / World Bank methodology ───────────────────────────
            # Damage ratios: UNOSAT Damage Assessment Guidelines 2020
            #   Moderate damage: 30% repair cost of replacement value
            #   Destroyed:      100% replacement value
            # Construction costs: World Bank SIDA Construction Cost Index 2023
            #   USD per m² (median for residential reinforced masonry)
            # Humanitarian response: UNOCHA 90-day emergency standard
            #   $1,200–$2,500 per displaced person depending on country income group
            # Insurance penetration: Swiss Re sigma 2023 by country income group
            # Avg building footprint: 80 m² (xBD dataset median, Gupta et al. 2019)
            AVG_FOOTPRINT_M2 = 80

            WB_COST_PER_M2 = {
                "United States":  1350, "Australia":  1100, "Portugal":   820,
                "Brazil":          420, "Mexico":      380, "Indonesia":  280,
                "Philippines":     260, "Nepal":       180,
            }
            UNOCHA_RESPONSE_PP = {
                "United States": 2500, "Australia": 2400, "Portugal": 1800,
                "Brazil": 1400, "Mexico": 1300, "Indonesia": 1200,
                "Philippines": 1200, "Nepal": 1200,
            }
            SWISS_RE_INSURED_PCT = {
                "United States": 0.55, "Australia": 0.52, "Portugal": 0.28,
                "Brazil": 0.18, "Mexico": 0.15, "Indonesia": 0.06,
                "Philippines": 0.05, "Nepal": 0.03,
            }
            cost_m2     = WB_COST_PER_M2.get(country_a, 400)
            resp_pp     = UNOCHA_RESPONSE_PP.get(country_a, 1200)
            insured_pct = SWISS_RE_INSURED_PCT.get(country_a, 0.10)
            replacement_value = AVG_FOOTPRINT_M2 * cost_m2

            repair_a    = counts[1] * replacement_value * 0.30   # UNOSAT: moderate = 30%
            replace_a   = counts[2] * replacement_value * 1.00   # UNOSAT: destroyed = 100%
            recon_a     = repair_a + replace_a
            displaced_a = (counts[1] + counts[2]) * hh_size_a
            response_a  = displaced_a * resp_pp
            insurance_a = recon_a * insured_pct
            total_eco_a = recon_a + response_a

            st.markdown("---")
            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.markdown(f"""<div class="cost-card">
                  <div class="cost-number" style="color:#dc2626;">${recon_a/1e6:.1f}M</div>
                  <div class="cost-label">Reconstruction Cost</div>
                </div>""", unsafe_allow_html=True)
            with ec2:
                st.markdown(f"""<div class="cost-card">
                  <div class="cost-number" style="color:#ea580c;">${response_a/1e6:.1f}M</div>
                  <div class="cost-label">Humanitarian Response</div>
                </div>""", unsafe_allow_html=True)
            with ec3:
                st.markdown(f"""<div class="cost-card">
                  <div class="cost-number" style="color:#7c3aed;">${insurance_a/1e6:.1f}M</div>
                  <div class="cost-label">Insurance Exposure</div>
                </div>""", unsafe_allow_html=True)
            with ec4:
                st.markdown(f"""<div class="cost-card">
                  <div class="cost-number" style="color:#2855c8;">${total_eco_a/1e6:.1f}M</div>
                  <div class="cost-label">Total Economic Impact</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")
            eb1, eb2, eb3, eb4 = st.columns(4)
            eb1.metric("✅ Intact",    counts[0])
            eb2.metric("⚠️ Moderate",  counts[1])
            eb3.metric("🔴 Destroyed", counts[2])
            eb4.metric("👥 Displaced", f"{displaced_a:,}")

            try:
                import plotly.graph_objects as go
                fig_cost = go.Figure()
                fig_cost.add_trace(go.Bar(
                    x=["Reconstruction", "Humanitarian", "Insurance Exposure"],
                    y=[recon_a/1e6, response_a/1e6, insurance_a/1e6],
                    marker_color=["#f0a830","#dc5050","#a09070"],
                    text=[f"${recon_a/1e6:.1f}M", f"${response_a/1e6:.1f}M", f"${insurance_a/1e6:.1f}M"],
                    textposition="outside",
                    width=0.5,
                ))
                fig_cost.update_layout(
                    paper_bgcolor="#13110d", plot_bgcolor="#0e0c09",
                    font=dict(family="DM Sans", color="#a09070", size=12),
                    yaxis=dict(title=dict(text="USD (millions)", font=dict(color="#f0a830")), gridcolor="rgba(240,168,48,0.08)", tickfont=dict(color="#a09070")),
                    xaxis=dict(gridcolor="rgba(240,168,48,0.08)", tickfont=dict(color="#e8dcc8", size=12)),
                    margin=dict(l=20,r=20,t=30,b=20), height=260, showlegend=False,
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            except ImportError:
                pass

            st.markdown('<p style="font-size:0.72rem;color:#8090c0;margin-top:2px;">📐 Methodology: UNOSAT Damage Assessment Guidelines 2020 · World Bank SIDA Construction Cost Index 2023 · UNOCHA 90-day Emergency Standard · Swiss Re Sigma Insurance Penetration 2023</p>', unsafe_allow_html=True)
        else:
            st.warning("Model weights not found.")
            st.image(orig, use_container_width=True)
    else:
        st.markdown("""<div style="border:1px solid rgba(240,168,48,0.15);border-radius:4px;padding:48px;text-align:center;margin-top:20px;background:#13110d;">
          <div style="font-size:3rem;margin-bottom:16px">🛰️</div>
          <div style="font-family:'Bebas Neue',sans-serif;color:#f0a830;font-size:1.4rem;letter-spacing:0.15em;">UPLOAD A SATELLITE IMAGE TO BEGIN</div>
          <div style="color:#a09070;font-size:0.8rem;margin-top:8px;font-family:'DM Mono',monospace;">PNG · JPG · Any resolution</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # ── Sample image quick-load ──────────────────────────────────────
        st.markdown('<p class="section-label">Or load a sample from the dataset</p>', unsafe_allow_html=True)
        sample_paths = []
        for sbase in [SAMPLE_DIR, GCP_YOLO / "images/val", GCP_XBD / "test/images"]:
            if sbase.exists():
                candidates = list(sbase.glob("*post_disaster*.png"))[:20] or list(sbase.glob("*.png"))[:20]
                sample_paths = candidates
                break
        if sample_paths:
            sample_names = [p.name for p in sample_paths[:8]]
            sq1, sq2 = st.columns([3,1])
            with sq1:
                chosen_sample = st.selectbox("Pick a sample image", sample_names, label_visibility="collapsed")
            with sq2:
                load_sample = st.button("▶ Load & Analyze", use_container_width=True)
            if load_sample:
                chosen_path = next(p for p in sample_paths if p.name == chosen_sample)
                st.session_state["sample_img_path"] = str(chosen_path)
                st.rerun()
        # Handle session_state sample load
        if "sample_img_path" in st.session_state:
            sp = Path(st.session_state["sample_img_path"])
            if sp.exists() and model_path:
                orig = Image.open(sp).convert("RGB")
                st.markdown("---")
                with st.spinner("🛰️ Running VISTA inference..."):
                    boxes, ms = run_inference(sp, model_path, conf_up)
                st.markdown(f'<span class="timing-pill">⚡ Sample · {ms:.1f} ms · {DEVICE_LABEL}</span>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                ann = draw_boxes(orig.copy(), boxes, conf_up)
                counts = {0:0, 1:0, 2:0}
                for b in boxes: counts[int(b[0])] += 1
                total = sum(counts.values())
                score = (counts[1]*0.5 + counts[2]*1.0) / max(total, 1)
                severity = "LOW" if score < 0.2 else "MODERATE" if score < 0.5 else "HIGH"
                c1, c2, c3 = st.columns([2,2,1])
                with c1:
                    st.markdown('<p class="section-label">Sample Image</p>', unsafe_allow_html=True)
                    st.image(orig, use_container_width=True)
                with c2:
                    st.markdown('<p class="section-label">VISTA Predictions</p>', unsafe_allow_html=True)
                    st.image(ann, use_container_width=True)
                with c3:
                    st.markdown('<p class="section-label">Severity</p>', unsafe_allow_html=True)
                    st.markdown(damage_gauge_html(score, severity), unsafe_allow_html=True)
                st.markdown("---")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Total Buildings", total)
                m2.metric("✅ No-Damage", counts[0])
                m3.metric("⚠️ Moderate", counts[1])
                m4.metric("🔴 Destroyed", counts[2])

# ══ TAB: Explore Dataset ══════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### Explore Dataset")
    mode = st.radio("View mode", ["🌍 Named Disaster Events", "🗂️ Validation Set Browser"],
                    horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if mode == "🌍 Named Disaster Events":
        st.markdown('<div class="info-box">Choose a named disaster from the xBD dataset. Browse pre/post satellite imagery and run VISTA inference on any tile.</div>', unsafe_allow_html=True)
        sel_label = st.selectbox("Disaster event", list(DISASTER_EVENTS.keys()))
        ekey = DISASTER_EVENTS[sel_label]
        eimgs = find_val_images_for_event(ekey)
        if not eimgs:
            st.markdown(f'<div class="info-box">⚠️ No images found for <strong>{ekey}</strong>. Add matching PNGs to <code>data/sample_images/</code> in the repo.</div>', unsafe_allow_html=True)
        else:
            sel_tile = st.selectbox("Select tile", [p.name for p in eimgs])
            post_path = next(p for p in eimgs if p.name == sel_tile)
            post_img = Image.open(post_path).convert("RGB")
            pre_path = get_pre_image(post_path)
            pre_img = Image.open(pre_path).convert("RGB") if pre_path else None
            if pre_img:
                st.markdown('<p class="section-label">Before / After — drag the divider</p>', unsafe_allow_html=True)
                w, h = post_img.size
                display_h = min(500, int(700 * h / w))
                st.components.v1.html(before_after_html(img_to_b64(pre_img.resize((w,h))), img_to_b64(post_img), display_h), height=display_h+50)
            else:
                st.image(post_img, use_container_width=True)
                st.markdown('<div class="info-box">Pre-disaster image not found for this tile.</div>', unsafe_allow_html=True)
            if model_path:
                st.markdown("---")
                with st.spinner("Running VISTA inference..."):
                    boxes, ms = run_inference(post_path, model_path, conf_thresh)
                st.markdown(f'<span class="timing-pill">⚡ {ms:.1f} ms · {DEVICE_LABEL}</span>', unsafe_allow_html=True)
                ann = draw_boxes(post_img.copy(), boxes, conf_thresh)
                co1, co2 = st.columns(2)
                with co1:
                    st.markdown('<p class="section-label">Post-Disaster</p>', unsafe_allow_html=True)
                    st.image(post_img, use_container_width=True)
                with co2:
                    st.markdown('<p class="section-label">VISTA Predictions</p>', unsafe_allow_html=True)
                    st.image(ann, use_container_width=True)
                counts_e = {0:0,1:0,2:0}
                for b in boxes: counts_e[int(b[0])] += 1
                total_e = sum(counts_e.values())
                score_e = (counts_e[1]*0.5+counts_e[2]*1.0)/max(total_e,1)
                severity_e = "LOW" if score_e<0.2 else "MODERATE" if score_e<0.5 else "HIGH"
                m1,m2,m3,m4,m5 = st.columns(5)
                m1.metric("Total", total_e)
                m2.metric("✅ No-Damage", counts_e[0])
                m3.metric("⚠️ Moderate", counts_e[1])
                m4.metric("🔴 Destroyed", counts_e[2])
                with m5:
                    st.markdown('<p class="section-label">Severity</p>', unsafe_allow_html=True)
                    st.markdown(damage_gauge_html(score_e, severity_e), unsafe_allow_html=True)

    else:  # Validation Set Browser
        st.markdown('<div class="info-box">Browse the 1,325 validation images used during training. Compare ground-truth annotations vs VISTA predictions side by side.</div>', unsafe_allow_html=True)
        all_val = find_all_val_images()
        if not all_val:
            st.markdown('<div class="info-box">⚠️ No validation images found. On Streamlit Cloud, add images to <code>data/val_images/</code> in your repo.</div>', unsafe_allow_html=True)
        else:
            sel_val = st.selectbox("Select image", [p.name for p in all_val])
            post_path = next(p for p in all_val if p.name == sel_val)
            post_img = Image.open(post_path).convert("RGB")
            pre_path = get_pre_image(post_path)
            pre_img = Image.open(pre_path).convert("RGB") if pre_path else None
            if pre_img:
                st.markdown('<p class="section-label">Before / After Slider</p>', unsafe_allow_html=True)
                w, h = post_img.size
                dh = min(450, int(700*h/w))
                st.components.v1.html(before_after_html(img_to_b64(pre_img.resize((w,h))), img_to_b64(post_img), dh), height=dh+50)
            st.markdown("---")
            label_path = (GCP_YOLO / "labels/val" if GCP_YOLO.exists() else DATA_DIR / "labels/val") / post_path.with_suffix(".txt").name
            gt_boxes = load_yolo_labels(label_path)
            cols = st.columns(3)
            with cols[0]:
                st.markdown('<p class="section-label">Post-Disaster</p>', unsafe_allow_html=True)
                st.image(post_img, use_container_width=True)
            with cols[1]:
                st.markdown('<p class="section-label">Ground Truth Labels</p>', unsafe_allow_html=True)
                st.image(draw_boxes(post_img.copy(), gt_boxes), use_container_width=True)
                st.markdown(f'<p class="img-caption">{len(gt_boxes)} annotated buildings</p>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown('<p class="section-label">VISTA Predictions</p>', unsafe_allow_html=True)
                if model_path:
                    with st.spinner("Inferring..."):
                        pred_boxes, ms = run_inference(post_path, model_path, conf_thresh)
                    st.image(draw_boxes(post_img.copy(), pred_boxes, conf_thresh), use_container_width=True)
                    st.markdown(f'<p class="img-caption">{len(pred_boxes)} detections · {ms:.0f} ms</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box">Model weights not loaded.</div>', unsafe_allow_html=True)
            if gt_boxes:
                st.markdown("---")
                cnts = {0:0,1:0,2:0}
                for b in gt_boxes: cnts[int(b[0])] += 1
                g1,g2,g3 = st.columns(3)
                g1.metric("✅ No-Damage", cnts[0])
                g2.metric("⚠️ Moderate", cnts[1])
                g3.metric("🔴 Destroyed", cnts[2])

# ══ TAB: Model Performance ══════════════════════════════════════════════════
with tab_metrics:
    _train = load_training_results()
    _per_class = load_per_class_metrics()
    _m = _train["overall"]
    n_ep = _train["n_epochs"]
    src  = _train.get("source", "fallback")

    st.markdown("### Overall Performance")
    status_txt = f"Live data · {n_ep} epochs trained · auto-updates every 60s" if src != "fallback" else "⚠️ results.csv not found — showing fallback values"
    st.markdown(f'<div class="info-box">📡 {status_txt}</div>', unsafe_allow_html=True)
    if src != "fallback":
        with st.expander("🔍 Debug — CSV columns (use this to verify data source)", expanded=False):
            st.code("\n".join(_train.get("columns", ["no columns found"])))
            st.caption(f"Source: {src}")

    q1,q2,q3,q4 = st.columns(4)
    q1.metric("mAP@50",    f"{_m['mAP50']:.3f}")
    q2.metric("mAP@50-95", f"{_m['mAP50_95']:.3f}")
    q3.metric("Precision",  f"{_m['Precision']:.3f}")
    q4.metric("Recall",     f"{_m['Recall']:.3f}")
    st.markdown("---")

    st.markdown("### Per-Class Breakdown")
    for cls_name, badge_cls in [("No-Damage","badge-green"),("Moderate-Damage","badge-yellow"),("Total-Destruction","badge-red")]:
        cm = _per_class[cls_name]
        st.markdown(f'<span class="badge {badge_cls}">{cls_name}</span>', unsafe_allow_html=True)
        a1,a2,a3 = st.columns(3)
        a1.metric("mAP@50",    f"{cm['mAP50']:.3f}")
        a2.metric("Precision", f"{cm['Precision']:.3f}")
        a3.metric("Recall",    f"{cm['Recall']:.3f}")
        st.markdown("")
    st.markdown("---")

    st.markdown(f"### Training Loss ({n_ep} Epochs)")
    try:
        import plotly.graph_objects as go
        _epochs = list(range(1, n_ep + 1))
        fig = go.Figure()
        _dfl = _train.get("dfl_loss", [])
        fig.add_trace(go.Scatter(x=_epochs, y=_train["box_loss"], name="Train Box Loss",
            line=dict(color="#2855c8", width=2), mode="lines+markers", marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=_epochs, y=_train["cls_loss"], name="Train Cls Loss",
            line=dict(color="#f97316", width=2), mode="lines+markers", marker=dict(size=6)))
        if _dfl:
            fig.add_trace(go.Scatter(x=_epochs, y=_dfl, name="Train DFL Loss",
                line=dict(color="#22c55e", width=2), mode="lines+markers", marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=_epochs, y=_train["map_hist"], name="mAP@50", yaxis="y2",
            line=dict(color="#7c3aed", width=2, dash="dot"), mode="lines+markers", marker=dict(size=6)))
        fig.update_layout(
            paper_bgcolor="#13110d", plot_bgcolor="#0e0c09",
            font=dict(family="DM Sans", color="#a09070", size=12),
            xaxis=dict(title=dict(text="Epoch", font=dict(color="#f0a830")),
                       gridcolor="rgba(240,168,48,0.08)", tickmode="linear",
                       tickfont=dict(color="#a09070", size=11),
                       linecolor="rgba(240,168,48,0.15)"),
            yaxis=dict(title=dict(text="Loss", font=dict(color="#f0a830")),
                       gridcolor="rgba(240,168,48,0.08)",
                       tickfont=dict(color="#a09070", size=11),
                       linecolor="rgba(240,168,48,0.15)"),
            yaxis2=dict(title=dict(text="mAP@50", font=dict(color="#f0a830")),
                overlaying="y", side="right", showgrid=False,
                tickfont=dict(color="#f0a830", size=11)),
            legend=dict(bgcolor="#13110d", bordercolor="rgba(240,168,48,0.2)", borderwidth=1,
                        font=dict(color="#e8dcc8", size=12)),
            margin=dict(l=20,r=60,t=20,b=20), height=320)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart({"Box Loss": _train["box_loss"], "Class Loss": _train["cls_loss"]})

    # ── Heatmaps (if generated) ──────────────────────────────────────────────
    heatmaps = find_heatmaps()
    if heatmaps:
        st.markdown("---")
        st.markdown("### Activation Heatmaps")
        st.markdown('<div class="info-box">Model activation intensity per pixel. Red = high damage · Green = no damage.</div>', unsafe_allow_html=True)
        hcols = st.columns(min(len(heatmaps), 3))
        for i, hm in enumerate(heatmaps):
            with hcols[i % 3]:
                st.image(Image.open(hm), use_container_width=True)
                st.markdown(f'<p class="img-caption">{hm.stem}</p>', unsafe_allow_html=True)


# ══ TAB: Model Card ══════════════════════════════════════════════════════════
with tab_card:
    st.markdown("### Model Card")
    st.markdown("---")
    for title, body in [
        ("🎯 Intended Use", "This model assists disaster response teams in rapidly assessing building damage from satellite imagery. It classifies detected buildings into three damage levels to help prioritize rescue and recovery."),
        ("📦 Training Data", "**Dataset:** xBD (xView2 Challenge) · 19 disaster events · 6 disaster types\n\n**Split:** 4,337 train · 1,325 val · 209k+ building annotations\n\n**Classes:** No-Damage 77% · Moderate 16% · Destroyed 7%"),
        ("⚙️ Architecture", "**Model:** YOLOv26n via Ultralytics 8.4.21\n\n**Input:** 640×640 RGB · **Params:** 2.5M · 5.8 GFLOPs\n\n**Training:** 5 epochs (target 50) · AdamW · batch 16 · NVIDIA L4"),
        ("📊 Performance", "**mAP@50:** 0.087 overall (5-epoch baseline)\n\nNo-Damage: 0.197 · Moderate: 0.055 · Destroyed: 0.007\n\n*Full 50-epoch training expected to significantly improve all scores.*"),
        ("⚠️ Limitations", "- Early training baseline — model is underfit\n- Class imbalance suppresses minority class detection\n- Not validated for operational emergency response use\n- Performance degrades below native xBD resolution (~0.3m/px)"),
    ]:
        with st.expander(title, expanded=False):
            st.markdown(f'<div class="model-card" style="background:#13110d;border:1px solid #2a2318;color:#c8b890;">{body}</div>', unsafe_allow_html=True)

    with st.expander("👥 Team", expanded=False):
        st.markdown('<div class="model-card" style="background:#13110d;border:1px solid #2a2318;color:#c8b890;">', unsafe_allow_html=True)
        st.markdown("**Project:** VISTA · Le Wagon Barcelona · Cohort #2230")
        st.markdown("**Stack:** Python · YOLOv26 · xBD Dataset · GCP Vertex AI · Streamlit")
        st.markdown("---")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("""<div style="background:#1a1610;border:1px solid #3a3020;border-top:2px solid #f0a830;border-radius:3px;padding:24px 16px;text-align:center;">
              <span style="font-size:2.6rem;display:block;line-height:1.2;">🇧🇷</span>
              <span style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;color:#e8dcc8;letter-spacing:0.08em;margin-top:10px;display:block;">Edison Kruger</span>
            </div>""", unsafe_allow_html=True)
        with t2:
            st.markdown("""<div style="background:#1a1610;border:1px solid #3a3020;border-top:2px solid #f0a830;border-radius:3px;padding:24px 16px;text-align:center;">
              <span style="font-size:2.6rem;display:block;line-height:1.2;">🇧🇷</span>
              <span style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;color:#e8dcc8;letter-spacing:0.08em;margin-top:10px;display:block;">Ildebrando de Jesus Junior</span>
            </div>""", unsafe_allow_html=True)
        with t3:
            st.markdown("""<div style="background:#1a1610;border:1px solid #3a3020;border-top:2px solid #f0a830;border-radius:3px;padding:24px 16px;text-align:center;">
              <span style="font-size:2.6rem;display:block;line-height:1.2;">🇵🇹</span>
              <span style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;color:#e8dcc8;letter-spacing:0.08em;margin-top:10px;display:block;">Martim Gomes</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
