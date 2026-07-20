"""
artifact4_geotiff_export.py
Exports predicted and reference road rasters as GeoTIFF per test municipality.
Uses TFW files to georeference each tile and mosaics them correctly.
Output: outputs/article/artifact4_geotiffs/
"""

import os

# ── Root paths — set environment variables before running ─────────────────
# Windows: set ROOT_DIR=D:\path	o\your\project
#          set ORTHO_DIR=\your
as\path	o\DOF025
# Linux:   export ROOT_DIR=/path/to/your/project
#          export ORTHO_DIR=/path/to/DOF025
_DEFAULT_ROOT  = r"D:\lstojanooad_extraction_slovenia"
_DEFAULT_ORTHO = r"\kgkn-nas\eo_data_2\GURS_podatki\DOF\DOF025"
ROOT_DIR  = Path(os.environ.get("ROOT_DIR",  _DEFAULT_ROOT))
ORTHO_DIR = Path(os.environ.get("ORTHO_DIR", _DEFAULT_ORTHO))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
TILE_INDEX_3  = ROOT_DIR / "data/processed/metadata/tile_index_road_only.csv"
MODELS_3CLASS = ROOT_DIR / "models"
TFW_ROOT      = ORTHO_DIR / "1_DOF025_Geolokacije/DOF025_TFW"
OUT_DIR       = ROOT_DIR / "outputs/article/artifact4_geotiffs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CRS — ESRI:102109 (ETRS_1989_Slovenia_TM)
EPSG_CODE  = 3794   # EPSG equivalent
PIXEL_SIZE = 0.25
TILE_SIZE  = 512

# ── Transform ──────────────────────────────────────────────────────────────
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── TFW cache ──────────────────────────────────────────────────────────────
tfw_cache = {}

def read_tfw(source_ortho):
    if source_ortho in tfw_cache:
        return tfw_cache[source_ortho]
    subfolder = source_ortho[0] + source_ortho[1:3]
    tfw_path  = TFW_ROOT / subfolder / f"{source_ortho}.tfw"
    if not tfw_path.exists():
        return None
    lines  = tfw_path.read_text().strip().split('\n')
    result = {'x_origin': float(lines[4]),
               'y_origin': float(lines[5])}
    tfw_cache[source_ortho] = result
    return result

def get_tile_geo(tile_row):
    tfw = read_tfw(tile_row['source_ortho'])
    if tfw is None:
        return None, None
    x = tfw['x_origin'] + tile_row['x_offset'] * PIXEL_SIZE
    y = tfw['y_origin'] + tile_row['y_offset'] * (-PIXEL_SIZE)
    return x, y

# ── Model ──────────────────────────────────────────────────────────────────
def load_segformer():
    model = smp.Segformer(encoder_name='mit_b2',
                          encoder_weights=None,
                          in_channels=3, classes=4).cuda()
    ckpt  = torch.load(
        MODELS_3CLASS / 'segformer' / 'segformer_best.pth',
        map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def predict(model, img):
    aug    = test_transform(image=img,
                            mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = aug['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(tensor)
    return out.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

# ── Mosaic builder ─────────────────────────────────────────────────────────
def build_mosaic(tile_rows, model=None, use_pred=False):
    """
    Build a geographic mosaic from a list of tiles.
    Returns (canvas_array, x_min, y_max) for GeoTIFF writing.
    """
    # Compute extent
    xs, ys = [], []
    for _, tile in tile_rows.iterrows():
        x, y = get_tile_geo(tile)
        if x is None:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        return None, None, None

    x_min = min(xs)
    y_max = max(ys)
    x_max = max(xs) + TILE_SIZE * PIXEL_SIZE
    y_min = min(ys) - TILE_SIZE * PIXEL_SIZE

    canvas_w = int(round((x_max - x_min) / PIXEL_SIZE))
    canvas_h = int(round((y_max - y_min) / PIXEL_SIZE))

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for _, tile in tile_rows.iterrows():
        x, y = get_tile_geo(tile)
        if x is None:
            continue
        col  = int(round((x - x_min) / PIXEL_SIZE))
        row_ = int(round((y_max - y) / PIXEL_SIZE))

        if use_pred and model is not None:
            img  = np.load(tile['image_path'])
            data = predict(model, img)
        else:
            data = np.load(tile['mask_path'])

        r0, r1 = row_, min(row_ + TILE_SIZE, canvas_h)
        c0, c1 = col,  min(col  + TILE_SIZE, canvas_w)
        canvas[r0:r1, c0:c1] = data[:r1-r0, :c1-c0]

    return canvas, x_min, y_max

def save_geotiff(canvas, x_min, y_max, out_path, crs_epsg=3794):
    transform = from_origin(x_min, y_max, PIXEL_SIZE, PIXEL_SIZE)
    crs       = CRS.from_epsg(crs_epsg)
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=canvas.shape[0],
        width=canvas.shape[1],
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(canvas, 1)
        # Write colormap
        # 0=background, 1=major, 2=local, 3=minor
        colormap = {
            0: (235, 235, 235, 255),
            1: (210, 40,  40,  255),
            2: (40,  90,  200, 255),
            3: (40,  170, 70,  255),
        }
        dst.write_colormap(1, colormap)
    print(f"    Saved: {out_path.name} "
          f"({canvas.shape[1]}×{canvas.shape[0]} px)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("ARTIFACT 4 — GEOTIFF EXPORT")
print("="*70)

tile_df_3 = pd.read_csv(TILE_INDEX_3)
test_df_3 = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)

municipalities = sorted(test_df_3['municipality'].unique())
print(f"Test municipalities: {municipalities}")

print("\nLoading SegFormer...")
segformer = load_segformer()
print("Loaded.")

geotiff_records = []

for muni in municipalities:
    print(f"\n--- {muni} ---")
    muni_tiles = test_df_3[test_df_3['municipality'] == muni]
    landscape  = muni_tiles['landscape_type'].iloc[0]
    print(f"  Tiles: {len(muni_tiles):,}")

    # Reference mask mosaic
    print(f"  Building reference mosaic...")
    ref_canvas, x_min, y_max = build_mosaic(
        muni_tiles, model=None, use_pred=False)

    if ref_canvas is None:
        print(f"  WARNING: Could not build mosaic — skipping")
        continue

    ref_path = OUT_DIR / f"{muni}_reference_3class.tif"
    save_geotiff(ref_canvas, x_min, y_max, ref_path, EPSG_CODE)

    # SegFormer prediction mosaic
    print(f"  Building SegFormer prediction mosaic...")
    pred_canvas, _, _ = build_mosaic(
        muni_tiles, model=segformer, use_pred=True)

    pred_path = OUT_DIR / f"{muni}_segformer_pred_3class.tif"
    save_geotiff(pred_canvas, x_min, y_max, pred_path, EPSG_CODE)

    # Quick connectivity stats
    road_ref  = (ref_canvas  > 0).sum()
    road_pred = (pred_canvas > 0).sum()
    total_px  = ref_canvas.size

    # Relaxed correctness/completeness (standard road metric)
    tp    = ((pred_canvas > 0) & (ref_canvas  > 0)).sum()
    fp    = ((pred_canvas > 0) & (ref_canvas == 0)).sum()
    fn    = ((pred_canvas == 0) & (ref_canvas > 0)).sum()
    correctness  = float(tp / (tp + fp + 1e-6))
    completeness = float(tp / (tp + fn + 1e-6))
    quality      = float(tp / (tp + fp + fn + 1e-6))

    print(f"  Road pixels — Ref: {road_ref:,} | "
          f"Pred: {road_pred:,}")
    print(f"  Correctness: {correctness:.4f} | "
          f"Completeness: {completeness:.4f} | "
          f"Quality: {quality:.4f}")

    geotiff_records.append({
        'municipality':  muni,
        'landscape':     landscape,
        'tiles':         len(muni_tiles),
        'canvas_w':      ref_canvas.shape[1],
        'canvas_h':      ref_canvas.shape[0],
        'road_ref_px':   int(road_ref),
        'road_pred_px':  int(road_pred),
        'correctness':   round(correctness,  4),
        'completeness':  round(completeness, 4),
        'quality':       round(quality,      4),
        'ref_tif':       ref_path.name,
        'pred_tif':      pred_path.name,
    })

summary_df = pd.DataFrame(geotiff_records)
summary_df.to_csv(OUT_DIR / 'artifact4_geotiff_summary.csv', index=False)

print(f"\n{'='*70}")
print(f"ALL GEOTIFFS SAVED TO: {OUT_DIR}")
print(f"{'='*70}")
print(summary_df[['municipality', 'landscape',
                   'correctness', 'completeness',
                   'quality']].to_string(index=False))
print("\nNote: GeoTIFFs are in EPSG:3794 (Slovenian national grid).")
print("Open in QGIS with the included colormap for visualization.")