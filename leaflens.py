#!/usr/bin/env python3
"""
LeafLens - desktop GUI for the five PlantVillage disease models (potato, tomato, strawberry, apple, corn).

Loads each crop's exported ONNX model + deploy_config.json (written by the training notebooks) and lets you
analyse photos from files, a laptop webcam or an ESP32-CAM, with multi-frame averaging, confidence thresholds,
an image-quality check, a history table and a feedback folder for building a real-camera dataset.

    pip install -r requirements.txt
    python leaflens.py                      # models folder is set in Settings (default: ./models)
    python leaflens.py --models "D:/models"
"""
import sys, os, json, csv, re, shutil, sqlite3, argparse, functools
import urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, Signal, QThread, QSize, QPoint, QRectF, QUrl
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPainterPath, QPen,
                           QPixmap)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu,
                               QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget, QDoubleSpinBox,
                               QAbstractItemView)

APP_NAME = "LeafLens"
APP_DIR = Path.home() / ".leaflens"
IMAGES_DIR = APP_DIR / "images"
DB_PATH = APP_DIR / "history.db"
MAX_FRAMES = 5

CROPS = [("potato", "Potato"), ("tomato", "Tomato"), ("strawberry", "Strawberry"), ("apple", "Apple"), ("corn", "Corn")]
CROP_TITLE = dict(CROPS)

# ESP32 CameraWebServer framesize enum values
ESP_SIZES = [("QVGA 320 x 240", 5), ("VGA 640 x 480", 8), ("SVGA 800 x 600", 9), ("XGA 1024 x 768", 10), ("SXGA 1280 x 1024", 12), ("UXGA 1600 x 1200", 13)]

DEFAULTS = dict(
    models_dir=str(Path(__file__).resolve().parent / "models"),
    feedback_dir=str(APP_DIR / "feedback"),
    conf_high=85, conf_low=50,
    esp_url="http://192.168.4.1/capture", esp_size=8, webcam_index=0,
    bg_removal=False, use_gpu=True, block_poor=True, min_sharpness=15.0, min_leaf_area=10,
    crop_mode="center_crop", use_calibration=False,
)

# ----------------------------------------------------------------------------------------------------------------
# Look & feel
# ----------------------------------------------------------------------------------------------------------------
BG, CARD, BORDER = "#FCEFD9", "#FFF7E6", "#3A4A38"
GREEN, GREEN_L, AMBER, RED, MUTED = "#2F7D4F", "#CDEBC3", "#F2C14E", "#D9534F", "#6E6B58"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", sans-serif; font-size: 13px; color: #2B2B22; }}
QMainWindow, QDialog, #root {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {CARD}; border: 2px solid {BORDER}; border-radius: 12px; }}
QFrame#inset {{ background: #FBEED3; border: 1px solid #E3D3AE; border-radius: 8px; }}
QLabel#h1 {{ font-size: 22px; font-weight: 800; color: #25402F; }}
QLabel#sub {{ color: {MUTED}; font-size: 12px; }}
QLabel#h2 {{ font-size: 14px; font-weight: 700; }}
QLabel#big {{ font-size: 24px; font-weight: 800; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QPushButton {{ background: {CARD}; border: 2px solid {BORDER}; border-radius: 8px; padding: 6px 8px; }}
QPushButton:hover {{ background: #F3E6C8; }}
QPushButton:checked {{ background: {GREEN_L}; font-weight: 600; }}
QPushButton:disabled {{ color: #A9A38C; border-color: #B9B49C; }}
QPushButton#primary {{ background: {GREEN}; color: white; font-weight: 800; font-size: 15px; padding: 11px; }}
QPushButton#primary:hover {{ background: #3A9262; }}
QPushButton#primary:disabled {{ background: #9DB8A6; border-color: #7E9486; color: #F4F4F4; }}
QPushButton#seg:checked {{ background: {GREEN}; color: white; }}
QPushButton#good {{ background: {GREEN}; color: white; font-weight: 700; }}
QPushButton#good:hover {{ background: #3A9262; }}
QPushButton#bad {{ background: {RED}; color: white; font-weight: 700; }}
QPushButton#bad:hover {{ background: #E26A66; }}
QPushButton#tab {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0; background: #F3E6C8; padding: 8px 12px; min-width: 120px; }}
QPushButton#tab:checked {{ background: {CARD}; font-weight: 700; }}
QPushButton#nav {{ border: 2px solid transparent; background: transparent; padding: 8px 4px; font-weight: 600; }}
QPushButton#nav:hover {{ background: #F3E6C8; }}
QPushButton#nav:checked {{ background: {GREEN_L}; border-color: {BORDER}; }}
QPushButton#chip {{ border: 1px solid {BORDER}; border-radius: 10px; padding: 3px 10px; background: {CARD}; font-size: 12px; }}
QToolButton {{ background: {CARD}; border: 2px solid {BORDER}; border-radius: 8px; padding: 3px 9px; }}
QToolButton:hover {{ background: #F3E6C8; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: white; border: 2px solid {BORDER}; border-radius: 8px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: white; selection-background-color: {GREEN_L}; selection-color: #222; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border: 2px solid {BORDER}; border-radius: 4px; background: white; }}
QCheckBox::indicator:checked {{ background: {GREEN}; }}
QSlider::groove:horizontal {{ height: 6px; background: #E3D3AE; border-radius: 3px; }}
QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; background: {CARD}; border: 2px solid {BORDER}; }}
QSlider#green::sub-page:horizontal {{ background: #4CAF6A; border-radius: 3px; }}
QSlider#amber::sub-page:horizontal {{ background: {AMBER}; border-radius: 3px; }}
QTableWidget {{ background: {CARD}; border: 2px solid {BORDER}; border-radius: 10px; gridline-color: #E3D3AE; selection-background-color: {GREEN_L}; selection-color: #222; }}
QHeaderView::section {{ background: #F3E6C8; border: none; border-bottom: 2px solid {BORDER}; padding: 7px; font-weight: 700; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #C9B88F; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""

PALETTE = {'k': "#3a2a1f", 'n': "#2b3a2e", 'g': "#58a65c", 'G': "#2f6b4f", 'l': "#9bd47a", 'r': "#d93a2b", 'R': "#a52418",
           'y': "#f2c744", 'Y': "#c9962b", 'b': "#b8854f", 'B': "#7a5230", 'w': "#fff3d6", 'o': "#e8893a", 's': "#f3e2bd",
           'p': "#e7708a", 'm': "#8b6b4a"}

GRIDS = {
    "leaf": [
        "...........nnnn.",
        ".........nnggggn",
        ".......nnggllgGn",
        "......nggllggGGn",
        ".....ngllgggGGGn",
        "....ngllggggGGn.",
        "....nglgggggGGn.",
        "...ngggggggGGn..",
        "...ngggggggGGn..",
        "..nggggggGGGn...",
        "..ngggggGGGn....",
        ".nggggGGGnn.....",
        ".nggGGGnn.......",
        "nkgGGnn.........",
        "nkGnn...........",
        "nn..............",
    ],
    "potato": [
        "............",
        "....nnnn....",
        "..nnbbbbnn..",
        ".nbbbwbbbbn.",
        ".nbbbbbBbbn.",
        "nbbBbbbbbbbn",
        "nbbbbbbbBbbn",
        "nbbbbwbbbbbn",
        ".nbbbbbbBbn.",
        ".nbBbbbbbbn.",
        "..nnbbbbnn..",
        "....nnnn....",
    ],
    "tomato": [
        ".....gG.....",
        "...gGgGGg...",
        "..nngGGgnn..",
        ".nrrrgGrrrn.",
        "nrrwrrrrrrrn",
        "nrwrrrrrrrRn",
        "nrrrrrrrrrRn",
        "nrrrrrrrrRRn",
        ".nrrrrrrRRn.",
        ".nRrrrrRRRn.",
        "..nnRRRRnn..",
        "....nnnn....",
    ],
    "strawberry": [
        "...gG.gG....",
        "....gGGg....",
        "..nngGGgnn..",
        ".nrrrrrrrrn.",
        "nrrwrrryrrRn",
        "nrrrrrrrrrRn",
        "nrryrrrwrrRn",
        ".nrrrrrrrRn.",
        ".nrrywrrRRn.",
        "..nrrrrRRn..",
        "...nrRRRn...",
        "....nnnn....",
    ],
    "apple": [
        ".......k....",
        "......kg gg.".replace(" ", ""),
        "...nnnkgGn..",
        "..nrrrnnrrn.",
        ".nrrrrrrrrrn",
        "nrrwrrrrrrRn",
        "nrwrrrrrrrRn",
        "nrrrrrrrrrRn",
        "nrrrrrrrrRRn",
        ".nrrrrrrRRn.",
        "..nnrrrRRnn.",
        "....nnnnn...",
    ],
    "corn": [
        "......nn....",
        "....nnyyn...",
        "...nyywyyn..",
        "..nywyyYyyn.",
        "..nyyyyyYyn.",
        "..nyYyyywyn.",
        "..nyyywyyyn.",
        ".ngnyyyyYyn.",
        ".nggnyyyyn..",
        "ngGggnnnn...",
        "nGGgn.......",
        ".nnn........",
    ],
    "home": [
        ".....nn.....",
        "....nnnn....",
        "...nnssnn...",
        "..nnssssnn..",
        ".nnssssssnn.",
        "nnssssssssnn",
        ".nssnnnnssn.",
        ".nssnbbnssn.",
        ".nssnbbnssn.",
        ".nnnnbbnnnn.",
        "............",
        "............",
    ],
    "history": [
        "...nnnnnn...",
        "..nssssssn..",
        ".nssssnsssn.",
        "nsssssnsssn.".replace("nsssssnsssn.", "nssssnsssssn")[:12],
        "nssssnsssssn",
        "nssssnnnnssn",
        "nsssssssssn.".replace("nsssssssssn.", "nssssssssssn")[:12],
        ".nssssssssn.",
        "..nssssssn..",
        "...nnnnnn...",
        "............",
        "............",
    ],
    "settings": [
        "....nnnn....",
        ".nn.nssn.nn.",
        ".nsnnssnnsn.",
        "..nssssssn..",
        "nnssnnnnssnn",
        "nssssnnssssn",
        "nssssnnssssn",
        "nnssnnnnssnn",
        "..nssssssn..",
        ".nsnnssnnsn.",
        ".nn.nssn.nn.",
        "....nnnn....",
    ],
    "sprout": [
        "......gg......",
        "....ggGGgg....",
        "...gGGggGGg...",
        "...gGgg.gGg...",
        "....gg...gg...",
        ".........gG...",
        "...gg....gG...",
        "..gGGg...gG...",
        "..gGGGg..gG...",
        "...gGGggGG....",
        ".....ggGG.....",
        "......gG......",
        "......mm......",
        ".....mBBm.....",
        "....mBBBBm....",
    ],
}


@functools.lru_cache(maxsize=None)
def pixmap(name, scale=2):
    rows = GRIDS[name]
    w = max(len(r) for r in rows)
    img = QImage(w, len(rows), QImage.Format_ARGB32)
    img.fill(0)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch in PALETTE:
                img.setPixelColor(x, y, QColor(PALETTE[ch]))
    return QPixmap.fromImage(img.scaled(w * scale, len(rows) * scale, Qt.IgnoreAspectRatio, Qt.FastTransformation))


# ----------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.path = APP_DIR / "settings.json"
        self.data = dict(DEFAULTS)
        try:
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            pass
        configured_models = Path(self.data["models_dir"])
        default_models = Path(DEFAULTS["models_dir"])
        if not configured_models.is_dir() and default_models.is_dir():
            self.data["models_dir"] = str(default_models)

    def __getitem__(self, k):
        return self.data.get(k, DEFAULTS.get(k))

    def get(self, k, default=None):
        return self.data.get(k, DEFAULTS.get(k, default))

    def set(self, k, v):
        self.data[k] = v
        self.save()

    def save(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def reset(self):
        keep = {k: self.data[k] for k in ("models_dir",) if k in self.data}
        self.data = dict(DEFAULTS); self.data.update(keep); self.save()


# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------
DEFAULT_CONFIGS = {
    "apple": {
        "crop": "Apple", "model": "efficientnet_b0",
        "class_names": ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy"],
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "temperature": 1.0, "conf_high": 0.85, "conf_low": 0.5, "test_accuracy": 1.0
    },
    "corn": {
        "crop": "Corn", "model": "efficientnet_b0",
        "class_names": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"],
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "temperature": 1.0, "conf_high": 0.85, "conf_low": 0.5, "test_accuracy": 0.9897
    },
    "potato": {
        "crop": "Potato", "model": "mobilenet_v2",
        "class_names": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "temperature": 1.0, "conf_high": 0.85, "conf_low": 0.5, "test_accuracy": 1.0
    },
    "strawberry": {
        "crop": "Strawberry", "model": "efficientnet_b0",
        "class_names": ["Strawberry___Leaf_scorch", "Strawberry___healthy"],
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "temperature": 1.0, "conf_high": 0.85, "conf_low": 0.5, "test_accuracy": 1.0
    },
    "tomato": {
        "crop": "Tomato", "model": "efficientnet_b0",
        "class_names": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"],
        "img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "temperature": 1.0, "conf_high": 0.85, "conf_low": 0.5, "test_accuracy": 0.9985
    }
}


def gpu_available():
    return "CUDAExecutionProvider" in ort.get_available_providers()


class ModelInfo:
    def __init__(self, crop, onnx_path, cfg):
        self.crop, self.onnx_path, self.cfg = crop, Path(onnx_path), cfg
        self.class_names = list(cfg["class_names"])
        self.img_size = int(cfg.get("img_size", 224))
        self.mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]), np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg.get("std", [0.229, 0.224, 0.225]), np.float32).reshape(3, 1, 1)
        self.temperature = float(cfg.get("temperature", 1.0))
        self.test_accuracy = cfg.get("test_accuracy")
        self._sess, self._sess_gpu = None, None

    def backend(self, use_gpu):
        return "ONNX Runtime (GPU)" if (use_gpu and gpu_available()) else "ONNX Runtime (CPU)"

    def session(self, use_gpu):
        want_gpu = bool(use_gpu and gpu_available())
        if self._sess is None or self._sess_gpu != want_gpu:
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if want_gpu else ["CPUExecutionProvider"]
            self._sess = ort.InferenceSession(str(self.onnx_path), providers=prov)
            self._sess_gpu = want_gpu
        return self._sess


def find_model(models_dir, crop):
    """Find <crop>'s .onnx + deploy_config.json under models_dir (flexible: models/potato/, PotatoDiseaseModel/, ...)."""
    root = Path(models_dir)
    if not root.is_dir():
        return None
    crop_lower = crop.lower()
    dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and crop_lower in d.name.lower()] + [root]
    for d in dirs:
        onnx_files = [p for p in sorted(d.glob("*.onnx")) if d != root or crop_lower in p.name.lower()]
        if not onnx_files:
            continue
        cfgs = [d / "deploy_config.json"] + sorted(d.glob(f"*{crop}*config*.json")) + sorted(d.glob("*config*.json"))
        for c in cfgs:
            if c.is_file():
                try:
                    text = c.read_text(encoding="utf-8")
                    # Try standard JSON first (dict or list)
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, dict):
                            if str(parsed.get("crop", "")).lower() == crop_lower:
                                return ModelInfo(crop, onnx_files[0], parsed)
                            elif crop_lower in parsed and isinstance(parsed[crop_lower], dict):
                                return ModelInfo(crop, onnx_files[0], parsed[crop_lower])
                        elif isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and str(item.get("crop", "")).lower() == crop_lower:
                                    return ModelInfo(crop, onnx_files[0], item)
                    except json.JSONDecodeError:
                        pass

                    # Fallback for concatenated JSON objects
                    decoder = json.JSONDecoder()
                    offset = 0
                    while offset < len(text):
                        while offset < len(text) and text[offset].isspace():
                            offset += 1
                        if offset >= len(text):
                            break
                        cfg, end = decoder.raw_decode(text, offset)
                        offset = end
                        if not isinstance(cfg, dict):
                            continue
                        if str(cfg.get("crop", crop)).lower() == crop_lower:
                            return ModelInfo(crop, onnx_files[0], cfg)
                except Exception:
                    continue
        # Fallback to default built-in configuration if config file is absent
        if crop_lower in DEFAULT_CONFIGS:
            return ModelInfo(crop, onnx_files[0], DEFAULT_CONFIGS[crop_lower])
    return None


def pretty_label(raw, crop):
    s = raw.split("___", 1)[1] if "___" in raw else raw
    s = re.sub(r"\s+", " ", s.replace("_", " ")).strip(" ()")
    if s.lower().startswith("spider mites"):
        return "Spider mites"
    if "gray leaf spot" in s.lower():
        return "Gray leaf spot"
    if s.lower().startswith(crop.lower() + " "):
        s = s[len(crop) + 1:]
    return (s[:1].upper() + s[1:]) if s else raw


GUIDE = [
    (("late blight",), "Water-soaked, pale green to dark brown or black patches that enlarge quickly; white fuzzy growth under the leaf in humid weather. Spreads fastest in cool, wet conditions.",
     "Act quickly: remove and bag infected plants or leaves (do not compost), avoid overhead watering, improve airflow, and ask your local extension service about a protectant fungicide."),
    (("early blight",), "Dark brown spots with concentric rings (a target look), often with a yellow halo, starting on older lower leaves.",
     "Remove affected lower leaves, mulch to stop soil splash, water at the base, rotate crops and keep plants well fed."),
    (("target spot",), "Brown lesions with faint concentric rings, sometimes with a yellow edge, on leaves and fruit; looks similar to early blight.",
     "Prune for airflow, avoid wetting foliage, remove infected debris, and follow local fungicide guidance if it keeps spreading."),
    (("bacterial spot",), "Small dark, water-soaked spots that turn scabby, often with a yellow halo; raised scabs on fruit.",
     "Use disease-free seed and transplants, avoid handling wet plants, rotate crops, and use copper-based sprays only as locally advised."),
    (("leaf mold",), "Pale green or yellow patches on the upper leaf surface with olive-green to grey velvety mold underneath. Common in humid, poorly ventilated spaces.",
     "Reduce humidity, ventilate or space plants, water at the base and remove infected leaves."),
    (("septoria",), "Many small round spots with dark borders and grey-white centres (tiny black dots inside), starting on lower leaves.",
     "Remove infected lower leaves, mulch, avoid overhead watering, rotate crops and clear plant debris after harvest."),
    (("spider mite",), "Fine yellow stippling, bronzing and dry leaves, with fine webbing on the underside. Worst in hot, dry weather.",
     "Rinse leaves with water, apply insecticidal soap or a miticide as advised locally, keep plants well watered and check neighbours."),
    (("yellow leaf curl", "curl virus"), "Upward-curling, yellowing and stunted leaves; plants fruit poorly. Spread by whiteflies.",
     "No cure: remove infected plants, control whiteflies (nets, sticky traps) and use resistant varieties next time."),
    (("mosaic",), "Light and dark green mottling, distorted or fern-like leaves and stunted growth.",
     "No cure: remove infected plants, wash hands and disinfect tools, do not use tobacco near plants, choose resistant varieties."),
    (("scab",), "Olive-green to brown velvety spots on leaves and rough, corky scabs on fruit.",
     "Rake and destroy fallen leaves, prune for airflow, and use a fungicide programme timed to spring wetness."),
    (("black rot",), "Brown 'frogeye' leaf spots with purple edges, rotting fruit that turns black, and cankers on branches.",
     "Prune out cankers and dead wood, remove mummified fruit, keep the tree canopy open and sanitise tools."),
    (("cedar", "rust"), "Bright yellow-orange spots on leaves (tiny black dots in the centre) with tube-like growths underneath later in the season.",
     "Remove nearby juniper or cedar hosts if possible, and use fungicide at the timing your local extension recommends."),
    (("scorch",), "Many small purple spots that enlarge and merge, centres turning grey-brown, so leaves look scorched.",
     "Remove infected leaves, avoid overhead watering, space plants for airflow, renovate beds after harvest and choose resistant varieties."),
    (("gray leaf spot", "grey leaf spot", "cercospora"), "Narrow rectangular tan to grey lesions running parallel to the veins, joining into large dead areas in humid weather.",
     "Rotate crops, bury or manage residue, choose resistant hybrids, and consider a fungicide if it appears early."),
    (("common rust",), "Small reddish-brown powdery pustules on both leaf surfaces, turning darker as the season ends.",
     "Plant resistant hybrids; fungicide is rarely needed unless infection is heavy before tasselling."),
    (("northern leaf blight", "leaf blight"), "Long cigar-shaped grey-green to tan lesions (several centimetres) that can cover whole leaves.",
     "Use resistant hybrids, rotate crops, manage residue, and consider a fungicide if it reaches upper leaves early."),
    (("healthy",), "No visible disease symptoms were detected on this leaf.",
     "Keep monitoring: water at the base, keep good airflow and remove any damaged leaves."),
]
GUIDE_DISCLAIMER = "General guidance only; confirm a diagnosis with your local agricultural extension service."


def guidance_for(label):
    low = label.lower()
    for keys, sym, act in GUIDE:
        if any(k in low for k in keys):
            return sym, act
    return "No stored description for this class.", "Check with your local agricultural extension service."


# ----------------------------------------------------------------------------------------------------------------
# Image processing
# ----------------------------------------------------------------------------------------------------------------
def load_rgb(path):
    try:
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.array(im.convert("RGB"))
            return limit_size(rgb)
    except Exception:
        data = np.fromfile(str(path), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        return limit_size(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def limit_size(rgb, max_side=1600):
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def qimage_from_rgb(rgb):
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def plant_mask(rgb):
    # Detect healthy greens/yellows as well as diseased browns/lesions
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green_yellow = (h >= 15) & (h <= 95) & (s >= 30) & (v >= 30)
    lesion_brown = (h >= 5) & (h <= 25) & (s >= 25) & (v >= 20)
    m = (green_yellow | lesion_brown).astype(np.uint8) * 255
    k = max(3, (min(rgb.shape[:2]) // 40) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    min_area = 0.01 * m.size
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            # Use convex hull to enclose interior necrotic spots and lesions
            hull = cv2.convexHull(c)
            cv2.drawContours(out, [hull], -1, 255, thickness=-1)
    return out


def remove_background(rgb):
    m = plant_mask(rgb)
    if m.mean() / 255 < 0.05:
        return rgb
    out = rgb.copy()
    out[m == 0] = 128
    return out


def assess_quality(rgb, st):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    if max(h, w) > 512:
        s = 512 / max(h, w)
        g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    small = rgb if max(h, w) <= 512 else cv2.resize(rgb, (int(w * 512 / max(h, w)), int(h * 512 / max(h, w))), interpolation=cv2.INTER_AREA)
    area = float(plant_mask(small).mean() / 255 * 100)
    problems = []
    if blur < st["min_sharpness"]:
        problems.append("too blurry")
    if area < st["min_leaf_area"]:
        problems.append("leaf too small or not found")
    return dict(blur=blur, area=area, ok=not problems, problems=problems)


def center_crop_square(pil_img):
    w, h = pil_img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return pil_img.crop((left, top, left + s, top + s))


def preprocess(rgb, info, bg_removal=False, crop_mode="center_crop"):
    if bg_removal:
        rgb = remove_background(rgb)
    s = getattr(info, "img_size", 224) if info else 224
    pil = Image.fromarray(rgb).convert("RGB")
    if crop_mode == "center_crop":
        pil = center_crop_square(pil)
    pil = pil.resize((s, s), Image.BILINEAR)
    return rgb, pil


def to_tensor(pil, info):
    pil = pil.convert("RGB")
    x = np.asarray(pil, np.float32).transpose(2, 0, 1) / 255.0
    return (x - info.mean) / info.std


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_inference(info, frames, bg_removal=False, use_gpu=True, crop_mode="center_crop", use_calibration=False):
    batch = []
    for f in frames:
        _, pil = preprocess(f, info, bg_removal, crop_mode)
        x = to_tensor(pil, info)
        batch.extend([x, np.ascontiguousarray(x[:, :, ::-1])])      # frame + its mirror image
    sess = info.session(use_gpu)
    temp = info.temperature if use_calibration else 1.0
    temp = max(temp, 0.05)
    logits = sess.run(None, {sess.get_inputs()[0].name: np.stack(batch).astype(np.float32)})[0] / temp
    return softmax(logits).mean(axis=0)


# ----------------------------------------------------------------------------------------------------------------
# Threads
# ----------------------------------------------------------------------------------------------------------------
class InferWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, info, frames, bg_removal, use_gpu, crop_mode="center_crop", use_calibration=False):
        super().__init__()
        self.info, self.frames, self.bg, self.gpu = info, frames, bg_removal, use_gpu
        self.crop_mode = crop_mode
        self.use_calibration = use_calibration

    def run(self):
        try:
            self.done.emit(run_inference(self.info, self.frames, self.bg, self.gpu, self.crop_mode, self.use_calibration))
        except Exception as e:                                       # noqa
            self.failed.emit(str(e))


def esp_base(url):
    p = urllib.parse.urlparse(url if "//" in url else "http://" + url)
    return f"{p.scheme}://{p.netloc}"


def esp_apply_size(url, val):
    try:
        urllib.request.urlopen(f"{esp_base(url)}/control?var=framesize&val={int(val)}", timeout=3).read()
    except Exception:
        pass


def esp_grab(url):
    data = urllib.request.urlopen(url if "//" in url else "http://" + url, timeout=5).read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The ESP32-CAM did not return a valid JPEG")
    return limit_size(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class GrabOnce(QThread):
    got = Signal(object)
    failed = Signal(str)

    def __init__(self, kind, st):
        super().__init__()
        self.kind, self.st = kind, dict(st.data)

    def run(self):
        try:
            if self.kind == "esp":
                esp_apply_size(self.st["esp_url"], self.st["esp_size"])
                self.got.emit(esp_grab(self.st["esp_url"]))
            else:
                cap = cv2.VideoCapture(int(self.st["webcam_index"]))
                if not cap.isOpened():
                    raise RuntimeError("Could not open the webcam (check the camera index in Settings)")
                frame = None
                for _ in range(8):
                    ok, f = cap.read()
                    if ok:
                        frame = f
                cap.release()
                if frame is None:
                    raise RuntimeError("The webcam returned no image")
                self.got.emit(limit_size(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        except Exception as e:                                       # noqa
            self.failed.emit(str(e))


class LiveSource(QThread):
    frame = Signal(object)
    error = Signal(str)

    def __init__(self, kind, st):
        super().__init__()
        self.kind, self.st, self._run = kind, dict(st.data), True

    def stop(self):
        self._run = False
        self.wait(3000)

    def run(self):
        cap = None
        try:
            if self.kind == "webcam":
                cap = cv2.VideoCapture(int(self.st["webcam_index"]))
                if not cap.isOpened():
                    self.error.emit("Could not open the webcam (check the camera index in Settings)")
                    return
                while self._run:
                    ok, f = cap.read()
                    if ok:
                        self.frame.emit(limit_size(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    self.msleep(30)
            else:
                esp_apply_size(self.st["esp_url"], self.st["esp_size"])
                while self._run:
                    try:
                        self.frame.emit(esp_grab(self.st["esp_url"]))
                    except Exception as e:                           # noqa
                        self.error.emit(f"ESP32-CAM: {e}")
                        self.msleep(1200)
                    self.msleep(120)
        finally:
            if cap is not None:
                cap.release()


# ----------------------------------------------------------------------------------------------------------------
# Small widgets
# ----------------------------------------------------------------------------------------------------------------
class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")


def pill(text, bg, fg="white", size=13):
    l = QLabel(text)
    l.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:10px; padding:3px 12px; font-weight:700; font-size:{size}px;")
    l.setAlignment(Qt.AlignCenter)
    return l


def set_pill(l, text, bg, fg="white", size=13):
    l.setText(text)
    l.setStyleSheet(f"background:{bg}; color:{fg}; border-radius:10px; padding:3px 12px; font-weight:700; font-size:{size}px;")


class ImageView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pm, self._zoom, self._off, self._drag, self.smooth = None, 1.0, QPoint(0, 0), None, True
        self.live = False
        self.placeholder = "Add a leaf photo to begin"

    def set_pixmap(self, pm, smooth=True, keep_view=False):
        self._pm, self.smooth = pm, smooth
        if not keep_view:
            self.fit()
        self.update()

    def fit(self):
        self._zoom, self._off = 1.0, QPoint(0, 0)
        self.update()

    def zoom_by(self, f):
        self._zoom = min(10.0, max(0.5, self._zoom * f))
        self.update()

    def wheelEvent(self, e):
        self.zoom_by(1.15 if e.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, e):
        self._drag = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            p = e.position().toPoint()
            self._off += p - self._drag
            self._drag = p
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath(); path.addRoundedRect(r, 10, 10)
        p.fillPath(path, QColor("#F6E9CC"))
        p.setClipPath(path)
        if self._pm is None or self._pm.isNull():
            p.setPen(QColor(MUTED)); p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
        else:
            pw, ph = self._pm.width(), self._pm.height()
            sc = min(self.width() / pw, self.height() / ph) * self._zoom
            w, h = pw * sc, ph * sc
            x = (self.width() - w) / 2 + self._off.x()
            y = (self.height() - h) / 2 + self._off.y()
            p.setRenderHint(QPainter.SmoothPixmapTransform, self.smooth)
            p.drawPixmap(QRectF(x, y, w, h), self._pm, QRectF(self._pm.rect()))
        p.setClipping(False)
        p.setPen(QPen(QColor("#CDB98A"), 1)); p.drawPath(path)
        if self.live:
            p.setBrush(QColor(RED)); p.setPen(Qt.NoPen); p.drawRoundedRect(12, 12, 54, 22, 8, 8)
            p.setPen(QColor("white")); p.setFont(QFont(self.font().family(), 9, QFont.Bold)); p.drawText(QRectF(12, 12, 54, 22), Qt.AlignCenter, "LIVE")


class FrameStrip(QWidget):
    selected = Signal(int)
    addRequested = Signal()
    removeRequested = Signal(int)

    def __init__(self):
        super().__init__()
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)
        self.set_frames([], -1)

    def set_frames(self, pms, sel):
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, pm in enumerate(pms):
            b = QToolButton()
            b.setIcon(QIcon(pm)); b.setIconSize(QSize(50, 50)); b.setFixedSize(60, 60)
            b.setToolTip("Click to view - right-click to remove")
            border = GREEN if i == sel else BORDER
            b.setStyleSheet(f"QToolButton {{ border: {3 if i == sel else 2}px solid {border}; border-radius: 8px; padding: 0; background:{CARD}; }}")
            b.clicked.connect(lambda _=False, i=i: self.selected.emit(i))
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda pos, i=i, b=b: self._menu(b, pos, i))
            self.lay.addWidget(b)
        if len(pms) < MAX_FRAMES:
            a = QToolButton(); a.setText("+"); a.setFixedSize(60, 60)
            a.setToolTip("Add a photo / capture a frame")
            a.setStyleSheet(f"QToolButton {{ border: 2px dashed #9A8E6C; border-radius: 8px; font-size: 22px; color:#9A8E6C; background: transparent; }}"
                            f"QToolButton:hover {{ background:#F3E6C8; }}")
            a.clicked.connect(self.addRequested)
            self.lay.addWidget(a)
        self.lay.addStretch(1)

    def _menu(self, b, pos, i):
        m = QMenu(self)
        act = m.addAction("Remove this frame")
        if m.exec(b.mapToGlobal(pos)) == act:
            self.removeRequested.emit(i)


class ProbBars(QWidget):
    ROW = 28

    def __init__(self):
        super().__init__()
        self.rows = []
        self.setMinimumHeight(self.ROW)

    def set_rows(self, rows):
        self.rows = rows
        self.setFixedHeight(max(self.ROW, len(rows) * self.ROW))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        lw, pw = min(170, int(self.width() * 0.38)), 46
        bx, bw = lw + 6, self.width() - lw - pw - 12
        for i, (name, prob) in enumerate(self.rows):
            y = i * self.ROW
            p.setPen(QColor("#2B2B22"))
            p.drawText(QRectF(0, y, lw, self.ROW), Qt.AlignVCenter | Qt.AlignLeft, self.fontMetrics().elidedText(name, Qt.ElideRight, lw))
            p.setPen(Qt.NoPen); p.setBrush(QColor("#E8DDBF"))
            p.drawRoundedRect(QRectF(bx, y + 8, bw, 12), 4, 4)
            p.setBrush(QColor("#3E9B5C") if i == 0 else QColor("#9DB9B3"))
            p.drawRoundedRect(QRectF(bx, y + 8, max(3.0, bw * prob), 12), 4, 4)
            p.setPen(QColor("#2B2B22"))
            p.drawText(QRectF(bx + bw + 6, y, pw, self.ROW), Qt.AlignVCenter | Qt.AlignLeft, f"{prob * 100:.0f}%")


class PixelDeco(QLabel):
    def __init__(self, name, scale=4):
        super().__init__()
        self.setPixmap(pixmap(name, scale))
        self.setAlignment(Qt.AlignCenter)


# ----------------------------------------------------------------------------------------------------------------
# History database
# ----------------------------------------------------------------------------------------------------------------
class HistoryDB:
    def __init__(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(DB_PATH))
        self.con.execute("""CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, crop TEXT, label TEXT, raw_label TEXT, confidence REAL,
            status TEXT, verdict TEXT, true_label TEXT, images TEXT, frames INTEGER)""")
        self.con.commit()

    def add(self, ts, crop, label, raw, conf, status, image_paths):
        cur = self.con.execute("INSERT INTO history(ts,crop,label,raw_label,confidence,status,verdict,true_label,images,frames) VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (ts, crop, label, raw, conf, status, "", "", json.dumps(image_paths), len(image_paths)))
        self.con.commit()
        return cur.lastrowid

    def rows(self, crop=None, limit=10, offset=0):
        q, a = "SELECT id,ts,crop,label,raw_label,confidence,status,verdict,true_label,images FROM history", []
        if crop:
            q += " WHERE crop=?"; a.append(crop)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        return self.con.execute(q, a + [limit, offset]).fetchall()

    def count(self, crop=None):
        if crop:
            return self.con.execute("SELECT COUNT(*) FROM history WHERE crop=?", (crop,)).fetchone()[0]
        return self.con.execute("SELECT COUNT(*) FROM history").fetchone()[0]

    def all(self):
        return self.con.execute("SELECT ts,crop,label,confidence,status,verdict,true_label,images FROM history ORDER BY id").fetchall()

    def set_verdict(self, rid, verdict, true_label):
        self.con.execute("UPDATE history SET verdict=?, true_label=? WHERE id=?", (verdict, true_label, rid))
        self.con.commit()


STATUS_TXT = {"confident": "Confident", "uncertain_retake": "Uncertain", "unknown": "Unknown"}
STATUS_COL = {"Confident": (GREEN, "white"), "Uncertain": (AMBER, "#3A2E00"), "Unknown": (RED, "white"),
              "Correct": (GREEN, "white"), "Wrong": (RED, "white")}


# ----------------------------------------------------------------------------------------------------------------
# Dialogs
# ----------------------------------------------------------------------------------------------------------------
class ModelStatusDialog(QDialog):
    def __init__(self, parent, crop, info, st):
        super().__init__(parent)
        self.setWindowTitle("Model Status")
        self.setMinimumWidth(480)
        v = QVBoxLayout(self); v.setContentsMargins(14, 14, 14, 14)
        card = Card(); v.addWidget(card)
        g = QGridLayout(card); g.setContentsMargins(18, 16, 18, 16); g.setHorizontalSpacing(24); g.setVerticalSpacing(10)
        t = QLabel("Model Status"); t.setObjectName("h2"); g.addWidget(t, 0, 0, 1, 2)
        rows = [("Crop", CROP_TITLE[crop])]
        if info:
            acc = info.test_accuracy
            rows += [("Model file", info.onnx_path.name),
                     ("Test accuracy", f"{acc * 100:.1f}%  (from deploy_config.json)" if isinstance(acc, (int, float)) else "not recorded"),
                     ("Classes", f"{len(info.class_names)}"), ("Input size", f"{info.img_size} x {info.img_size}"),
                     ("Calibration T", f"{info.temperature:.3f}" + (" (active)" if st.get("use_calibration", False) else " (inactive, T=1.0 raw)")),
                     ("Backend", info.backend(st["use_gpu"]))]
        else:
            rows += [("Model file", "not found"), ("Looked in", st["models_dir"])]
        for i, (a, b) in enumerate(rows, start=1):
            la = QLabel(a); la.setObjectName("muted"); lb = QLabel(b); lb.setWordWrap(True)
            g.addWidget(la, i, 0); g.addWidget(lb, i, 1)
        ok = pill("Ready", GREEN) if info else pill("No model", RED)
        g.addWidget(ok, len(rows) + 1, 1, alignment=Qt.AlignRight)
        b = QPushButton("Close"); b.clicked.connect(self.accept); v.addWidget(b, alignment=Qt.AlignRight)


# ----------------------------------------------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------------------------------------------
class HistoryPage(QWidget):
    def __init__(self, db, st, models_ref):
        super().__init__()
        self.db, self.st, self.models_ref = db, st, models_ref
        self.page, self.cache, self.cur = 0, {}, None
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)

        top = QHBoxLayout()
        t = QLabel("History"); t.setObjectName("h2"); top.addWidget(t)
        top.addStretch(1)
        top.addWidget(QLabel("Crop:"))
        self.crop_filter = QComboBox(); self.crop_filter.addItem("All crops", None)
        for k, n in CROPS:
            self.crop_filter.addItem(n, k)
        self.crop_filter.currentIndexChanged.connect(lambda: self._goto(0))
        top.addWidget(self.crop_filter)
        v.addLayout(top)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Time", "Crop", "Label", "Confidence", "Status", "Thumbnail"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.Fixed); self.table.setColumnWidth(5, 90)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(56)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setIconSize(QSize(44, 44))
        self.table.itemSelectionChanged.connect(self._selected)
        v.addWidget(self.table, 1)

        bar = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV"); self.export_btn.clicked.connect(self.export_csv); bar.addWidget(self.export_btn)
        bar.addStretch(1)
        self.prev_btn = QPushButton("<"); self.next_btn = QPushButton(">")
        self.prev_btn.clicked.connect(lambda: self._goto(self.page - 1)); self.next_btn.clicked.connect(lambda: self._goto(self.page + 1))
        self.page_lbl = QLabel("Page 1"); self.page_lbl.setObjectName("muted")
        bar.addWidget(self.prev_btn); bar.addWidget(self.page_lbl); bar.addWidget(self.next_btn)
        bar.addSpacing(14); bar.addWidget(QLabel("Items per page:"))
        self.per = QComboBox()
        for n in (10, 25, 50):
            self.per.addItem(str(n), n)
        self.per.currentIndexChanged.connect(lambda: self._goto(0))
        bar.addWidget(self.per)
        v.addLayout(bar)

        self.detail = Card()
        d = QHBoxLayout(self.detail); d.setContentsMargins(14, 10, 14, 10)
        self.d_img = QLabel(); self.d_img.setFixedSize(76, 76); self.d_img.setAlignment(Qt.AlignCenter)
        d.addWidget(self.d_img)
        col = QVBoxLayout()
        self.d_title = QLabel("Select an entry"); self.d_title.setObjectName("h2")
        self.d_sub = QLabel(""); self.d_sub.setObjectName("muted")
        col.addWidget(self.d_title); col.addWidget(self.d_sub); d.addLayout(col, 1)
        self.ok_btn = QPushButton("Mark as correct"); self.ok_btn.setObjectName("good")
        self.bad_btn = QPushButton("Mark as wrong"); self.bad_btn.setObjectName("bad")
        self.ok_btn.clicked.connect(self.mark_correct); self.bad_btn.clicked.connect(self.mark_wrong)
        b = QVBoxLayout(); b.addWidget(self.ok_btn); b.addWidget(self.bad_btn); d.addLayout(b)
        self.fb_note = QLabel(""); self.fb_note.setObjectName("muted")
        v.addWidget(self.detail)
        v.addWidget(self.fb_note)
        self._set_detail(None)
        self.refresh()

    # -- data
    def _thumb(self, path, size=44):
        key = (path, size)
        if key not in self.cache:
            pm = QPixmap(path)
            self.cache[key] = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation) if not pm.isNull() else QPixmap()
        return self.cache[key]

    def _goto(self, p):
        total = self.db.count(self.crop_filter.currentData())
        pages = max(1, -(-total // self.per.currentData()))
        self.page = min(max(0, p), pages - 1)
        self.refresh()

    def refresh(self):
        crop, per = self.crop_filter.currentData(), self.per.currentData()
        total = self.db.count(crop)
        pages = max(1, -(-total // per))
        self.page = min(self.page, pages - 1)
        rows = self.db.rows(crop, per, self.page * per)
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        self.row_ids = {}
        for r, (rid, ts, c, label, raw, conf, status, verdict, true_label, images) in enumerate(rows):
            self.row_ids[r] = rid
            self.table.setItem(r, 0, QTableWidgetItem(ts))
            it = QTableWidgetItem(QIcon(pixmap(c, 2)), "  " + CROP_TITLE.get(c, c)); self.table.setItem(r, 1, it)
            self.table.setItem(r, 2, QTableWidgetItem(label))
            self.table.setItem(r, 3, QTableWidgetItem(f"{conf * 100:.0f}%"))
            txt = (verdict.capitalize() if verdict else STATUS_TXT.get(status, status))
            bg, fg = STATUS_COL.get(txt, (MUTED, "white"))
            holder = QWidget(); hl = QHBoxLayout(holder); hl.setContentsMargins(6, 0, 6, 0)
            hl.addWidget(pill(txt, bg, fg, 12), alignment=Qt.AlignLeft | Qt.AlignVCenter)
            self.table.setCellWidget(r, 4, holder)
            imgs = json.loads(images) if images else []
            th = QLabel(); th.setAlignment(Qt.AlignCenter)
            if imgs:
                th.setPixmap(self._thumb(imgs[0]))
            self.table.setCellWidget(r, 5, th)
        self.table.blockSignals(False)
        self.page_lbl.setText(f"Page {self.page + 1} / {pages}   ({total} entries)")
        self.prev_btn.setEnabled(self.page > 0); self.next_btn.setEnabled(self.page < pages - 1)
        self.cur = None
        self._set_detail(None)

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._set_detail(None); return
        r = rows[0].row()
        rid = self.row_ids.get(r)
        full = self.db.con.execute("SELECT id,ts,crop,label,raw_label,confidence,status,verdict,true_label,images FROM history WHERE id=?", (rid,)).fetchone()
        self.cur = full
        self._set_detail(full)

    def _set_detail(self, row):
        has = row is not None
        self.ok_btn.setEnabled(has); self.bad_btn.setEnabled(has)
        if not has:
            self.d_img.clear(); self.d_title.setText("Select an entry"); self.d_sub.setText("Then mark it correct or wrong to save it to your feedback folder.")
            return
        rid, ts, crop, label, raw, conf, status, verdict, true_label, images = row
        imgs = json.loads(images) if images else []
        if imgs:
            self.d_img.setPixmap(self._thumb(imgs[0], 72))
        self.d_title.setText(f"{CROP_TITLE.get(crop, crop)}: {label}   {conf * 100:.0f}%")
        extra = f"   |   marked {verdict}" + (f" (true: {pretty_label(true_label, crop)})" if true_label and verdict == "wrong" else "") if verdict else ""
        self.d_sub.setText(f"{ts}   |   {len(imgs)} frame(s){extra}")

    # -- feedback
    def _save_feedback(self, crop, class_raw):
        dest = Path(self.st["feedback_dir"]) / crop / class_raw
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in json.loads(self.cur[9]):
            if os.path.exists(p):
                shutil.copy2(p, dest / Path(p).name); n += 1
        return dest, n

    def mark_correct(self):
        if not self.cur:
            return
        rid, crop, raw = self.cur[0], self.cur[2], self.cur[4]
        dest, n = self._save_feedback(crop, raw)
        self.db.set_verdict(rid, "correct", raw)
        self.fb_note.setText(f"Saved {n} image(s) to {dest}")
        self._keep_refresh(rid)

    def mark_wrong(self):
        if not self.cur:
            return
        rid, crop = self.cur[0], self.cur[2]
        info = self.models_ref().get(crop)
        if not info:
            QMessageBox.information(self, APP_NAME, "No model is loaded for this crop, so the class list is unknown.")
            return
        names = [pretty_label(c, crop) for c in info.class_names]
        choice, ok = QInputDialog.getItem(self, "Mark as wrong", "What is the correct class?", names, 0, False)
        if not ok:
            return
        raw = info.class_names[names.index(choice)]
        dest, n = self._save_feedback(crop, raw)
        self.db.set_verdict(rid, "wrong", raw)
        self.fb_note.setText(f"Saved {n} image(s) as '{choice}' to {dest}")
        self._keep_refresh(rid)

    def _keep_refresh(self, rid):
        self.refresh()
        for r, i in self.row_ids.items():
            if i == rid:
                self.table.selectRow(r); break

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export history", "leaflens_history.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "crop", "label", "confidence", "status", "user_verdict", "true_label", "image_paths"])
            for ts, crop, label, conf, status, verdict, true_label, images in self.db.all():
                w.writerow([ts, crop, label, f"{conf:.4f}", STATUS_TXT.get(status, status), verdict, true_label, ";".join(json.loads(images or "[]"))])
        self.fb_note.setText(f"Exported to {path}")


class SettingsPage(QWidget):
    modelsChanged = Signal()
    changed = Signal()

    def __init__(self, st, models_ref):
        super().__init__()
        self.st, self.models_ref = st, models_ref
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setStyleSheet("background: transparent;")
        outer.addWidget(sc)
        inner = QWidget(); inner.setStyleSheet("background: transparent;"); sc.setWidget(inner)
        v = QVBoxLayout(inner); v.setContentsMargins(2, 2, 10, 2); v.setSpacing(12)

        # --- models
        c1, l1 = self._section("Model & Thresholds")
        row = QHBoxLayout(); row.addWidget(QLabel("Models folder"))
        self.models_edit = QLineEdit(st["models_dir"]); self.models_edit.editingFinished.connect(self._models_edited)
        br = QPushButton("Browse..."); br.clicked.connect(self._browse_models)
        row.addWidget(self.models_edit, 1); row.addWidget(br); l1.addLayout(row)
        self.found_lbl = QLabel(""); self.found_lbl.setObjectName("muted"); self.found_lbl.setWordWrap(True); l1.addWidget(self.found_lbl)
        self.s_high, self.v_high = self._slider(l1, "Confident threshold (green)", "green", 50, 99, st["conf_high"], "conf_high", "%d%%")
        self.s_low, self.v_low = self._slider(l1, "Uncertain threshold (amber)", "amber", 10, 95, st["conf_low"], "conf_low", "%d%%")
        self.red_lbl = QLabel(); self.red_lbl.setObjectName("muted"); l1.addWidget(self.red_lbl)
        self._update_red()
        self.calib = QCheckBox("Enable temperature calibration (Warning: sharpens confidence)")
        self.calib.setChecked(st.get("use_calibration", False))
        self.calib.toggled.connect(lambda x: self._set("use_calibration", x))
        l1.addWidget(self.calib)
        v.addWidget(c1)

        # --- esp32 + webcam
        c2, l2 = self._section("Cameras")
        g = QGridLayout(); g.setVerticalSpacing(8); l2.addLayout(g)
        g.addWidget(QLabel("ESP32-CAM capture URL"), 0, 0)
        self.esp_url = QLineEdit(st["esp_url"]); self.esp_url.editingFinished.connect(lambda: self._set("esp_url", self.esp_url.text().strip()))
        g.addWidget(self.esp_url, 0, 1)
        g.addWidget(QLabel("ESP32 resolution"), 1, 0)
        self.esp_size = QComboBox()
        for n, val in ESP_SIZES:
            self.esp_size.addItem(n, val)
        self.esp_size.setCurrentIndex(max(0, self.esp_size.findData(st["esp_size"])))
        self.esp_size.currentIndexChanged.connect(lambda: self._set("esp_size", self.esp_size.currentData()))
        g.addWidget(self.esp_size, 1, 1)
        g.addWidget(QLabel("Webcam index"), 2, 0)
        self.cam = QSpinBox(); self.cam.setRange(0, 9); self.cam.setValue(st["webcam_index"])
        self.cam.valueChanged.connect(lambda x: self._set("webcam_index", x))
        g.addWidget(self.cam, 2, 1)
        note = QLabel("Resolution is sent to the camera via /control?var=framesize (standard ESP32 CameraWebServer firmware); other firmware may ignore it.")
        note.setObjectName("muted"); note.setWordWrap(True); l2.addWidget(note)
        v.addWidget(c2)

        # --- processing
        c3, l3 = self._section("Image processing")
        row_crop = QHBoxLayout(); row_crop.addWidget(QLabel("Aspect ratio mode"))
        self.crop_mode = QComboBox()
        self.crop_mode.addItem("Center-crop square (preserves leaf proportions)", "center_crop")
        self.crop_mode.addItem("Stretch to square (legacy)", "stretch")
        crop_idx = max(0, self.crop_mode.findData(st.get("crop_mode", "center_crop")))
        self.crop_mode.setCurrentIndex(crop_idx)
        self.crop_mode.currentIndexChanged.connect(lambda: self._set("crop_mode", self.crop_mode.currentData()))
        row_crop.addWidget(self.crop_mode, 1); l3.addLayout(row_crop)
        self.bg = QCheckBox("Apply leaf background removal (experimental)")
        self.bg.setChecked(st["bg_removal"]); self.bg.toggled.connect(lambda x: self._set("bg_removal", x)); l3.addWidget(self.bg)
        n = QLabel("Only helps if your models were trained on images processed the same way; otherwise leave it off."); n.setObjectName("muted"); n.setWordWrap(True); l3.addWidget(n)
        self.block = QCheckBox("Warn before analysing blurry or leaf-less images")
        self.block.setChecked(st["block_poor"]); self.block.toggled.connect(lambda x: self._set("block_poor", x)); l3.addWidget(self.block)
        g3 = QGridLayout(); l3.addLayout(g3)
        g3.addWidget(QLabel("Minimum sharpness (Laplacian variance)"), 0, 0)
        self.sharp = QDoubleSpinBox(); self.sharp.setRange(0, 1000); self.sharp.setDecimals(1); self.sharp.setValue(st["min_sharpness"])
        self.sharp.valueChanged.connect(lambda x: self._set("min_sharpness", x)); g3.addWidget(self.sharp, 0, 1)
        g3.addWidget(QLabel("Minimum leaf area (%)"), 1, 0)
        self.area = QSpinBox(); self.area.setRange(0, 90); self.area.setValue(int(st["min_leaf_area"]))
        self.area.valueChanged.connect(lambda x: self._set("min_leaf_area", x)); g3.addWidget(self.area, 1, 1)
        n2 = QLabel("Tip: look at the Blur value under the image for a few sharp and blurry shots from your camera, then set the minimum between them."); n2.setObjectName("muted"); n2.setWordWrap(True); l3.addWidget(n2)
        v.addWidget(c3)

        # --- advanced
        c4, l4 = self._section("Advanced")
        self.gpu = QCheckBox("Use GPU (if available)" + ("" if gpu_available() else " - CUDA not found, will use CPU"))
        self.gpu.setChecked(st["use_gpu"]); self.gpu.toggled.connect(lambda x: self._set("use_gpu", x)); l4.addWidget(self.gpu)
        r4 = QHBoxLayout(); r4.addWidget(QLabel("Feedback folder"))
        self.fb = QLineEdit(st["feedback_dir"]); self.fb.editingFinished.connect(lambda: self._set("feedback_dir", self.fb.text().strip()))
        ob = QPushButton("Open"); ob.clicked.connect(self._open_fb)
        r4.addWidget(self.fb, 1); r4.addWidget(ob); l4.addLayout(r4)
        rb = QPushButton("Reset to defaults"); rb.clicked.connect(self._reset); l4.addWidget(rb, alignment=Qt.AlignLeft)
        v.addWidget(c4)
        v.addStretch(1)
        self.refresh_found()

    def _section(self, title):
        c = Card(); l = QVBoxLayout(c); l.setContentsMargins(16, 12, 16, 14); l.setSpacing(8)
        t = QLabel(title); t.setObjectName("h2"); l.addWidget(t)
        return c, l

    def _slider(self, layout, text, name, lo, hi, val, key, fmt):
        r = QHBoxLayout(); lab = QLabel(text); lab.setMinimumWidth(190)
        s = QSlider(Qt.Horizontal); s.setObjectName(name); s.setRange(lo, hi); s.setValue(int(val))
        v = QLabel(fmt % val); v.setMinimumWidth(40)
        s.valueChanged.connect(lambda x, key=key, v=v, fmt=fmt: self._slider_changed(key, x, v, fmt))
        r.addWidget(lab); r.addWidget(s, 1); r.addWidget(v); layout.addLayout(r)
        return s, v

    def _slider_changed(self, key, x, label, fmt):
        label.setText(fmt % x)
        if key == "conf_high" and x - 5 < self.st["conf_low"]:
            self.s_low.setValue(max(self.s_low.minimum(), x - 5))
        if key == "conf_low" and x + 5 > self.st["conf_high"]:
            self.s_high.setValue(min(self.s_high.maximum(), x + 5))
        self._set(key, x)
        self._update_red()

    def _update_red(self):
        self.red_lbl.setText(f"Unknown (red): below {self.st['conf_low']}%")

    def _set(self, k, v):
        self.st.set(k, v); self.changed.emit()

    def _browse_models(self):
        d = QFileDialog.getExistingDirectory(self, "Select the folder that contains your models", self.st["models_dir"])
        if d:
            self.models_edit.setText(d); self._models_edited()

    def _models_edited(self):
        self.st.set("models_dir", self.models_edit.text().strip()); self.modelsChanged.emit(); self.refresh_found()

    def refresh_found(self):
        infos = self.models_ref()
        parts = []
        for k, n in CROPS:
            i = infos.get(k)
            parts.append(f"{'OK' if i else 'missing'}  {n}" + (f" ({i.onnx_path.name})" if i else ""))
        self.found_lbl.setText("   |   ".join(parts))

    def _open_fb(self):
        Path(self.st["feedback_dir"]).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.st["feedback_dir"]))

    def _reset(self):
        self.st.reset()
        for w, k in ((self.s_high, "conf_high"), (self.s_low, "conf_low")):
            w.setValue(int(self.st[k]))
        if hasattr(self, "calib"):
            self.calib.setChecked(self.st.get("use_calibration", False))
        if hasattr(self, "crop_mode"):
            idx = max(0, self.crop_mode.findData(self.st.get("crop_mode", "center_crop")))
            self.crop_mode.setCurrentIndex(idx)
        self.esp_url.setText(self.st["esp_url"]); self.esp_size.setCurrentIndex(self.esp_size.findData(self.st["esp_size"]))
        self.cam.setValue(self.st["webcam_index"]); self.bg.setChecked(self.st["bg_removal"]); self.block.setChecked(self.st["block_poor"])
        self.sharp.setValue(self.st["min_sharpness"]); self.area.setValue(int(self.st["min_leaf_area"]))
        self.gpu.setChecked(self.st["use_gpu"]); self.fb.setText(self.st["feedback_dir"]); self.changed.emit()


# ----------------------------------------------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self, st):
        super().__init__()
        self.setWindowTitle("LeafLens - Plant Disease Detector")
        self.setWindowIcon(QIcon(pixmap("leaf", 4)))
        self.setObjectName("root")
        self.resize(1320, 800); self.setMinimumSize(1150, 700)
        self.setAcceptDrops(True)
        self.st, self.db = st, HistoryDB()
        self.infos = {}
        self.crop, self.frames, self.sel = "tomato", [], -1
        self.live, self.live_frame, self.grabber, self.worker = None, None, None, None
        self._threads = set()
        self.view_tab, self.result, self.show_all = "original", None, False
        self.refresh_models()
        self._build()
        self._select_crop("tomato")

    def _run_thread(self, t):
        """Keep a reference until the thread has really finished (avoids 'QThread destroyed while running')."""
        self._threads.add(t)
        t.finished.connect(lambda t=t: self._threads.discard(t))
        t.start()

    # -------------------------------------------------- models
    def refresh_models(self):
        self.infos = {k: find_model(self.st["models_dir"], k) for k, _ in CROPS}

    # -------------------------------------------------- UI build
    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(14, 10, 14, 8); root.setSpacing(8)

        head = QHBoxLayout()
        logo = QLabel(); logo.setPixmap(pixmap("leaf", 3)); head.addWidget(logo)
        tv = QVBoxLayout(); tv.setSpacing(0)
        t = QLabel(APP_NAME); t.setObjectName("h1"); s = QLabel("Plant Disease Detector"); s.setObjectName("sub")
        tv.addWidget(t); tv.addWidget(s); head.addLayout(tv)
        head.addStretch(1)
        self.page_title = QLabel("Home"); self.page_title.setObjectName("h2"); head.addWidget(self.page_title)
        root.addLayout(head)

        body = QHBoxLayout(); body.setSpacing(12); root.addLayout(body, 1)

        # sidebar
        side = QVBoxLayout(); side.setSpacing(6)
        self.nav = QButtonGroup(self); self.nav.setExclusive(True)
        for i, (key, text) in enumerate((("home", "Home"), ("history", "History"), ("settings", "Settings"))):
            b = QPushButton(text); b.setObjectName("nav"); b.setCheckable(True); b.setFixedWidth(104)
            b.setIcon(QIcon(pixmap(key, 3))); b.setIconSize(QSize(36, 36))
            b.setStyleSheet("padding:6px 2px; font-size:12px;")
            b.clicked.connect(lambda _=False, i=i: self._goto_page(i))
            self.nav.addButton(b, i); side.addWidget(b)
        self.nav.button(0).setChecked(True)
        side.addStretch(1); side.addWidget(PixelDeco("sprout", 5))
        sw = QWidget(); sw.setLayout(side); sw.setFixedWidth(112); body.addWidget(sw)

        self.stack = QStackedWidget(); body.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_home())
        self.history = HistoryPage(self.db, self.st, lambda: self.infos); self.stack.addWidget(self.history)
        self.settings_page = SettingsPage(self.st, lambda: self.infos); self.stack.addWidget(self.settings_page)
        self.settings_page.modelsChanged.connect(self._models_changed)
        self.settings_page.changed.connect(self._settings_changed)

        # status bar
        bar = QHBoxLayout()
        self.chip = QPushButton(); self.chip.setObjectName("chip"); self.chip.setCursor(Qt.PointingHandCursor)
        self.chip.clicked.connect(self._show_model_status)
        bar.addWidget(self.chip)
        self.ready = pill("Ready", GREEN, size=11); bar.addWidget(self.ready)
        bar.addStretch(1)
        self.notice = QLabel(""); self.notice.setObjectName("muted"); bar.addWidget(self.notice)
        root.addLayout(bar)

    def _build_home(self):
        page = QWidget()
        h = QHBoxLayout(page); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(12)

        # ---- left: inputs
        left = Card(); left.setFixedWidth(410)
        lv = QVBoxLayout(left); lv.setContentsMargins(16, 14, 16, 14); lv.setSpacing(10)
        lv.addWidget(self._h2("1.  Select your crop"))
        grid = QGridLayout(); grid.setSpacing(8)
        self.crop_group = QButtonGroup(self); self.crop_group.setExclusive(True)
        for i, (k, n) in enumerate(CROPS):
            b = QPushButton(" " + n); b.setCheckable(True); b.setIcon(QIcon(pixmap(k, 2))); b.setIconSize(QSize(24, 24))
            b.setMinimumHeight(40)
            b.clicked.connect(lambda _=False, k=k: self._select_crop(k, user=True))
            self.crop_group.addButton(b); b.setProperty("crop", k)
            grid.addWidget(b, i // 3, i % 3)
        lv.addLayout(grid)
        info = QFrame(); info.setObjectName("inset"); il = QHBoxLayout(info); il.setContentsMargins(10, 6, 10, 6)
        il.addWidget(QLabel("(i)  This model only knows its own crop's classes - pick the plant you are scanning."))
        il.itemAt(0).widget().setObjectName("muted"); il.itemAt(0).widget().setWordWrap(True)
        lv.addWidget(info)

        lv.addWidget(self._h2("2.  Choose input source"))
        seg = QHBoxLayout(); seg.setSpacing(6)
        self.src_group = QButtonGroup(self); self.src_group.setExclusive(True)
        for i, n in enumerate(("Upload", "Webcam", "ESP32-CAM")):
            b = QPushButton(n); b.setObjectName("seg"); b.setCheckable(True); b.setMinimumHeight(40)
            self.src_group.addButton(b, i); seg.addWidget(b)
        self.src_group.button(0).setChecked(True)
        self.src_group.idClicked.connect(self._source_changed)
        lv.addLayout(seg)

        self.src_stack = QStackedWidget()
        up = QWidget(); ul = QHBoxLayout(up); ul.setContentsMargins(0, 0, 0, 0)
        self.choose_btn = QPushButton("Choose image(s)..."); self.choose_btn.clicked.connect(self.choose_files)
        ul.addWidget(self.choose_btn); hint = QLabel("or drag & drop here"); hint.setObjectName("muted"); ul.addWidget(hint, 1)
        lv_ = QWidget(); ll = QHBoxLayout(lv_); ll.setContentsMargins(0, 0, 0, 0)
        self.preview_btn = QPushButton("Start preview"); self.preview_btn.setCheckable(True); self.preview_btn.clicked.connect(self._toggle_preview)
        self.capture_btn = QPushButton("Capture frame"); self.capture_btn.clicked.connect(self.capture)
        ll.addWidget(self.preview_btn); ll.addWidget(self.capture_btn)
        self.src_stack.addWidget(up); self.src_stack.addWidget(lv_)
        lv.addWidget(self.src_stack)

        fr = QFrame(); fr.setObjectName("inset"); fl = QVBoxLayout(fr); fl.setContentsMargins(10, 8, 10, 10)
        fh = QHBoxLayout(); self.frames_lbl = QLabel(); self.frames_lbl.setObjectName("h2"); fh.addWidget(self.frames_lbl); fh.addStretch(1)
        self.trash = QToolButton(); self.trash.setText("Clear"); self.trash.setToolTip("Remove all frames"); self.trash.clicked.connect(self.clear_frames); fh.addWidget(self.trash)
        fl.addLayout(fh)
        self.strip = FrameStrip()
        self.strip.selected.connect(self.select_frame); self.strip.addRequested.connect(self._add_clicked); self.strip.removeRequested.connect(self.remove_frame)
        fl.addWidget(self.strip)
        tip = QLabel("Several frames of the same leaf from slightly different angles are averaged for a steadier result."); tip.setObjectName("muted"); tip.setWordWrap(True); fl.addWidget(tip)
        lv.addWidget(fr)

        self.analyze_btn = QPushButton("Analyze"); self.analyze_btn.setObjectName("primary"); self.analyze_btn.clicked.connect(self.analyze)
        lv.addWidget(self.analyze_btn)
        self.left_note = QLabel(""); self.left_note.setObjectName("muted"); self.left_note.setWordWrap(True); lv.addWidget(self.left_note)
        lv.addStretch(1)
        h.addWidget(left)

        # ---- middle: image
        mid = Card()
        mv = QVBoxLayout(mid); mv.setContentsMargins(14, 12, 14, 12); mv.setSpacing(0)
        tabs = QHBoxLayout(); tabs.setSpacing(2)
        self.tab_orig = QPushButton("Original"); self.tab_prep = QPushButton("Preprocessed")
        for b in (self.tab_orig, self.tab_prep):
            b.setObjectName("tab"); b.setCheckable(True)
        self.tab_orig.setChecked(True)
        self.tab_orig.clicked.connect(lambda: self._set_tab("original")); self.tab_prep.clicked.connect(lambda: self._set_tab("prep"))
        tabs.addWidget(self.tab_orig); tabs.addWidget(self.tab_prep); tabs.addStretch(1)
        mv.addLayout(tabs)
        self.view = ImageView(); mv.addSpacing(6); mv.addWidget(self.view, 1)
        zr = QHBoxLayout(); zr.setContentsMargins(0, 8, 0, 8)
        for txt, f in (("  -  ", lambda: self.view.zoom_by(1 / 1.25)), ("  +  ", lambda: self.view.zoom_by(1.25)), ("Fit", self.view.fit)):
            b = QToolButton(); b.setText(txt); b.clicked.connect(f); zr.addWidget(b)
        zr.addStretch(1); mv.addLayout(zr)
        self.q_box = QFrame(); self.q_box.setObjectName("inset")
        ql = QHBoxLayout(self.q_box); ql.setContentsMargins(12, 8, 12, 8)
        self.q_icon = QLabel("?"); self.q_icon.setFixedSize(26, 26); self.q_icon.setAlignment(Qt.AlignCenter)
        ql.addWidget(self.q_icon)
        qv = QVBoxLayout(); qv.setSpacing(0)
        self.q_title = QLabel("No image yet"); self.q_title.setObjectName("h2")
        self.q_sub = QLabel(""); self.q_sub.setObjectName("muted")
        qv.addWidget(self.q_title); qv.addWidget(self.q_sub); ql.addLayout(qv, 1)
        mv.addWidget(self.q_box)
        h.addWidget(mid, 3)

        # ---- right: result
        right = Card(); right.setMinimumWidth(380)
        rv = QVBoxLayout(right); rv.setContentsMargins(18, 16, 18, 14); rv.setSpacing(10)
        self.empty_lbl = QLabel("Your analysis will appear here.\n\nAdd a leaf photo, then press Analyze.")
        self.empty_lbl.setAlignment(Qt.AlignCenter); self.empty_lbl.setObjectName("muted"); self.empty_lbl.setWordWrap(True)
        rv.addWidget(self.empty_lbl, 1)

        self.res = QWidget(); r = QVBoxLayout(self.res); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(10)
        top = QHBoxLayout(); self.res_icon = QLabel(); top.addWidget(self.res_icon)
        tv = QVBoxLayout(); tv.setSpacing(4)
        self.res_title = QLabel(""); self.res_title.setObjectName("big"); self.res_title.setWordWrap(True)
        self.res_badge = pill("", GREEN, size=14)
        tv.addWidget(self.res_title); tv.addWidget(self.res_badge, alignment=Qt.AlignLeft)
        top.addLayout(tv, 1); r.addLayout(top)
        self.res_hint = QLabel(""); self.res_hint.setWordWrap(True); self.res_hint.setObjectName("muted"); r.addWidget(self.res_hint)
        self.bars = ProbBars(); r.addWidget(self.bars)
        self.all_btn = QPushButton(); self.all_btn.setFlat(True); self.all_btn.setCursor(Qt.PointingHandCursor)
        self.all_btn.setStyleSheet("QPushButton { border:none; color:#2F7D4F; font-weight:600; text-align:left; padding:0; } QPushButton:hover { background: transparent; text-decoration: underline; }")
        self.all_btn.clicked.connect(self._toggle_all); r.addWidget(self.all_btn, alignment=Qt.AlignLeft)
        self.sym_box = self._info_box("Symptoms"); self.act_box = self._info_box("Suggested action")
        r.addWidget(self.sym_box[0]); r.addWidget(self.act_box[0])
        self.res_meta = QLabel(""); self.res_meta.setObjectName("muted"); self.res_meta.setWordWrap(True); r.addWidget(self.res_meta)
        r.addStretch(1)
        self.res_scroll = QScrollArea(); self.res_scroll.setWidgetResizable(True); self.res_scroll.setFrameShape(QFrame.NoFrame)
        self.res_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.res_scroll.setStyleSheet("QScrollArea { background: transparent; } QScrollArea > QWidget > QWidget { background: transparent; }")
        self.res_scroll.setWidget(self.res)
        rv.addWidget(self.res_scroll, 1)
        self.res_scroll.hide()
        h.addWidget(right, 2)
        return page

    def _h2(self, t):
        l = QLabel(t); l.setObjectName("h2"); return l

    def _info_box(self, title):
        f = QFrame(); f.setObjectName("inset"); l = QVBoxLayout(f); l.setContentsMargins(12, 8, 12, 10); l.setSpacing(3)
        t = QLabel(title); t.setObjectName("h2"); body = QLabel(""); body.setWordWrap(True)
        l.addWidget(t); l.addWidget(body)
        return f, body

    # -------------------------------------------------- navigation / settings
    def _goto_page(self, i):
        self.stack.setCurrentIndex(i)
        self.page_title.setText(("Home", "History", "Settings")[i])
        if i == 1:
            self.history.refresh()
        if i != 0:
            self._stop_live()

    def _models_changed(self):
        self.refresh_models(); self._update_status(); self.settings_page.refresh_found()

    def _settings_changed(self):
        self._update_status()

    def _show_model_status(self):
        ModelStatusDialog(self, self.crop, self.infos.get(self.crop), self.st).exec()

    def _update_status(self):
        info = self.infos.get(self.crop)
        if info:
            acc = info.test_accuracy
            self.chip.setText(f"{info.onnx_path.name}   |   " + (f"test acc {acc * 100:.1f}%   |   " if isinstance(acc, (int, float)) else "") + info.backend(self.st["use_gpu"]))
            set_pill(self.ready, "Ready", GREEN, size=11)
        else:
            self.chip.setText(f"No {CROP_TITLE[self.crop]} model found in {self.st['models_dir']}")
            set_pill(self.ready, "No model", RED, size=11)
        self.analyze_btn.setEnabled(self.worker is None)

    # -------------------------------------------------- crop / source
    def _select_crop(self, crop, user=False):
        changed = crop != self.crop
        self.crop = crop
        for b in self.crop_group.buttons():
            b.setChecked(b.property("crop") == crop)
        if user and changed and self.frames:
            self.left_note.setText(f"Crop changed to {CROP_TITLE[crop]} - make sure these photos show a {CROP_TITLE[crop].lower()} leaf, then press Analyze again.")
        elif user:
            self.left_note.setText("")
        if user and changed:
            self._clear_result()
        self._update_status()
        self._refresh_view()

    def _source_changed(self, i):
        self._stop_live()
        self.src_stack.setCurrentIndex(0 if i == 0 else 1)
        if i == 1:
            self.preview_btn.setText("Start preview"); self.capture_btn.setText("Capture frame")
        elif i == 2:
            self.preview_btn.setText("Start preview"); self.capture_btn.setText("Capture frame")
            self.left_note.setText(f"ESP32-CAM URL: {self.st['esp_url']}  (change in Settings)")
        else:
            self.left_note.setText("")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    # -------------------------------------------------- frames
    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose leaf images", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        self._add_paths(paths)

    def _add_paths(self, paths):
        for p in paths:
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                try:
                    if not self.add_frame(load_rgb(p)):
                        break
                except Exception as e:                               # noqa
                    self.notice.setText(str(e))

    def _add_clicked(self):
        if self.src_group.checkedId() == 0:
            self.choose_files()
        else:
            self.capture()

    def add_frame(self, rgb):
        if len(self.frames) >= MAX_FRAMES:
            self.notice.setText(f"Maximum of {MAX_FRAMES} frames - remove one first.")
            return False
        self.frames.append(rgb)
        self.sel = len(self.frames) - 1
        self._clear_result()
        self._refresh_strip(); self._refresh_view()
        return True

    def remove_frame(self, i):
        if 0 <= i < len(self.frames):
            del self.frames[i]
            self.sel = min(self.sel, len(self.frames) - 1)
            self._clear_result(); self._refresh_strip(); self._refresh_view()

    def clear_frames(self):
        self.frames, self.sel = [], -1
        self._clear_result(); self._refresh_strip(); self._refresh_view()

    def select_frame(self, i):
        self.sel = i
        self._refresh_strip(); self._refresh_view()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.stack.currentIndex() == 0 and self.sel >= 0:
            self.remove_frame(self.sel)
        else:
            super().keyPressEvent(e)

    def _refresh_strip(self):
        pms = [QPixmap.fromImage(qimage_from_rgb(f)).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation) for f in self.frames]
        self.strip.set_frames(pms, self.sel)
        self.frames_lbl.setText(f"Captured frames ({len(self.frames)}/{MAX_FRAMES})")

    # -------------------------------------------------- live sources
    def _toggle_preview(self, on):
        if on:
            kind = "webcam" if self.src_group.checkedId() == 1 else "esp"
            self.live = LiveSource(kind, self.st)
            self.live.frame.connect(self._on_live_frame); self.live.error.connect(lambda m: self.notice.setText(m))
            self.live.start()
            self.preview_btn.setText("Stop preview")
        else:
            self._stop_live()

    def _stop_live(self):
        if self.live:
            self.live.stop(); self.live = None
        self.live_frame = None
        self.view.live = False
        if hasattr(self, "preview_btn"):
            self.preview_btn.setChecked(False); self.preview_btn.setText("Start preview")
        self._refresh_view()

    def _on_live_frame(self, rgb):
        self.live_frame = rgb
        if self.view_tab == "original" and self.stack.currentIndex() == 0:
            self.view.live = True
            self.view.set_pixmap(QPixmap.fromImage(qimage_from_rgb(rgb)), keep_view=True)

    def capture(self):
        kind = "webcam" if self.src_group.checkedId() == 1 else "esp"
        if len(self.frames) >= MAX_FRAMES:
            self.notice.setText(f"Maximum of {MAX_FRAMES} frames - remove one first."); return
        if self.live and self.live_frame is not None:
            self.add_frame(self.live_frame.copy()); return
        self.capture_btn.setEnabled(False); self.notice.setText("Capturing...")
        self.grabber = GrabOnce(kind, self.st)
        self.grabber.got.connect(self._grab_done); self.grabber.failed.connect(self._grab_failed)
        self._run_thread(self.grabber)

    def _grab_done(self, rgb):
        self.capture_btn.setEnabled(True); self.notice.setText("")
        self.add_frame(rgb)

    def _grab_failed(self, msg):
        self.capture_btn.setEnabled(True); self.notice.setText(msg)
        QMessageBox.warning(self, APP_NAME, msg)

    # -------------------------------------------------- viewing
    def _set_tab(self, name):
        self.view_tab = name
        self.tab_orig.setChecked(name == "original"); self.tab_prep.setChecked(name == "prep")
        self._refresh_view()

    def _refresh_view(self):
        info = self.infos.get(self.crop)
        size = info.img_size if info else 224
        self.tab_prep.setText(f"Preprocessed {size}x{size}")
        if self.live_frame is not None and self.live and self.view_tab == "original":
            return
        self.view.live = False
        if not (0 <= self.sel < len(self.frames)):
            self.view.placeholder = "Add a leaf photo to begin"
            self.view.set_pixmap(None); self._set_quality(None); return
        rgb = self.frames[self.sel]
        if self.view_tab == "original":
            self.view.set_pixmap(QPixmap.fromImage(qimage_from_rgb(rgb)))
        else:
            pre, pil = preprocess(rgb, info or type("I", (), {"img_size": 224})(), self.st["bg_removal"], self.st.get("crop_mode", "center_crop"))
            arr = np.asarray(pil)
            self.view.set_pixmap(QPixmap.fromImage(qimage_from_rgb(np.ascontiguousarray(arr))), smooth=False)
        self._set_quality(assess_quality(rgb, self.st))

    def _set_quality(self, q):
        if q is None:
            self.q_icon.setText("?"); self.q_icon.setStyleSheet(f"background:#CFC6A8; color:white; border-radius:13px; font-weight:800;")
            self.q_title.setText("No image yet"); self.q_sub.setText(""); return
        if q["ok"]:
            self.q_icon.setText("OK"); self.q_icon.setStyleSheet(f"background:{GREEN}; color:white; border-radius:13px; font-weight:800; font-size:10px;")
            self.q_title.setText("Image quality looks good!")
        else:
            self.q_icon.setText("!"); self.q_icon.setStyleSheet(f"background:{AMBER}; color:#3A2E00; border-radius:13px; font-weight:800;")
            self.q_title.setText("Check this image: " + ", ".join(q["problems"]))
        self.q_sub.setText(f"Blur: {q['blur']:.1f}     |     Leaf area: {q['area']:.0f}%")

    # -------------------------------------------------- analysis
    def _clear_result(self):
        self.result = None
        self.res_scroll.hide(); self.empty_lbl.show()

    def analyze(self):
        if not self.frames:
            self.left_note.setText("Add at least one leaf photo first."); return
        info = self.infos.get(self.crop)
        if not info:
            QMessageBox.warning(self, APP_NAME, f"No {CROP_TITLE[self.crop]} model found.\n\nLooked in:\n{self.st['models_dir']}\n\nSet the models folder in Settings.")
            return
        quals = [assess_quality(f, self.st) for f in self.frames]
        poor_indices = [i for i, q in enumerate(quals) if not q["ok"]]
        note = ""
        use = list(self.frames)
        if poor_indices and self.st["block_poor"]:
            probs = sorted({p for i in poor_indices for p in quals[i]["problems"]})
            box = QMessageBox(self); box.setWindowTitle(APP_NAME); box.setIcon(QMessageBox.Warning)
            box.setText(f"{len(poor_indices)} of {len(self.frames)} frame(s) have potential quality issues (" + ", ".join(probs) + ").")
            box.setInformativeText("Results may be less reliable. You can analyse anyway or retake.")
            retake = box.addButton("Retake", QMessageBox.RejectRole)
            anyway = box.addButton("Analyse anyway", QMessageBox.AcceptRole)
            box.exec()
            if box.clickedButton() != anyway:
                return
            note = f"Analysed with quality flags: {', '.join(probs)}."
        elif poor_indices:
            note = f"{len(poor_indices)} frame(s) had image quality warnings."
        self._used_frames = use; self._note = note
        self.analyze_btn.setEnabled(False); self.analyze_btn.setText("Analyzing...")
        self.worker = InferWorker(info, use, self.st["bg_removal"], self.st["use_gpu"],
                                  crop_mode=self.st.get("crop_mode", "center_crop"),
                                  use_calibration=self.st.get("use_calibration", False))
        self.worker.done.connect(lambda probs, info=info: self._analysis_done(info, probs))
        self.worker.failed.connect(self._analysis_failed)
        self._run_thread(self.worker)

    def _analysis_failed(self, msg):
        self.worker = None; self.analyze_btn.setEnabled(True); self.analyze_btn.setText("Analyze")
        QMessageBox.critical(self, APP_NAME, f"Analysis failed:\n{msg}")

    def _analysis_done(self, info, probs):
        self.worker = None; self.analyze_btn.setEnabled(True); self.analyze_btn.setText("Analyze")
        top = int(np.argmax(probs)); conf = float(probs[top])
        status = "confident" if conf >= self.st["conf_high"] / 100 else ("uncertain_retake" if conf >= self.st["conf_low"] / 100 else "unknown")
        order = np.argsort(-probs)
        self.result = dict(crop=self.crop, raw=info.class_names[top], label=pretty_label(info.class_names[top], self.crop), conf=conf, status=status,
                           probs=[(pretty_label(info.class_names[i], self.crop), float(probs[i])) for i in order], frames=len(self._used_frames), note=self._note)
        self._save_history(info)
        self._show_result()

    def _save_history(self, info):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = []
        for i, f in enumerate(self._used_frames):
            p = IMAGES_DIR / f"{stamp}_{self.crop}_{i}.jpg"
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(str(p)); paths.append(str(p))
        r = self.result
        self.db.add(ts, self.crop, r["label"], r["raw"], r["conf"], r["status"], paths)

    def _toggle_all(self):
        self.show_all = not self.show_all
        self._show_result()

    def _show_result(self):
        r = self.result
        if not r:
            return
        self.empty_lbl.hide(); self.res_scroll.show()
        self.res_icon.setPixmap(pixmap(r["crop"], 5))
        pct = f"{r['conf'] * 100:.0f}%"
        if r["status"] == "confident":
            self.res_title.setText(f"{CROP_TITLE[r['crop']]}: {r['label']}")
            set_pill(self.res_badge, f"Confident   {pct}", GREEN, size=14); self.res_hint.setText("")
        elif r["status"] == "uncertain_retake":
            self.res_title.setText(f"{CROP_TITLE[r['crop']]}: {r['label']}?")
            set_pill(self.res_badge, f"Uncertain   {pct}", AMBER, "#3A2E00", size=14)
            self.res_hint.setText("Uncertain prediction - retake: move closer, avoid glare, keep leaf centered, or inspect runner-up classes below.")
        else:
            self.res_title.setText("Unknown")
            set_pill(self.res_badge, f"Low confidence   {pct}", RED, size=14)
            self.res_hint.setText(f"Best guess was {r['label']}, but confidence is low. Verify crop selection and compare candidate classes below.")
        rows = r["probs"] if self.show_all else r["probs"][:5]
        self.bars.set_rows(rows)
        if len(r["probs"]) > 5:
            self.all_btn.setText("Show top 5" if self.show_all else f"Show all {len(r['probs'])} classes"); self.all_btn.show()
        else:
            self.all_btn.hide()
        sym, act = guidance_for(r["label"])
        self.sym_box[1].setText(sym); self.act_box[1].setText(act + "\n\n" + GUIDE_DISCLAIMER)
        meta = f"Averaged over {r['frames']} frame(s) and their mirror images."
        self.res_meta.setText(meta + ("  " + r["note"] if r["note"] else ""))

    def closeEvent(self, e):
        self._stop_live()
        for t in list(self._threads):
            t.wait(3000)
        super().closeEvent(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="folder containing the exported models (overrides Settings for this run)")
    args, qt_args = ap.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    app.setStyleSheet(QSS)
    st = Settings()
    if args.models:
        st.data["models_dir"] = args.models
    w = MainWindow(st)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
