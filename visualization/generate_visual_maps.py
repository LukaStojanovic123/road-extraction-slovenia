"""
generate_visual_maps.py
Generates publication-quality visual maps for ISPRS article.
- Finds best center tile per test municipality
- Stitches 3x3 spatially contiguous grid using exact TFW geographic coordinates
- Generates combined figures for 3-class and 2-class
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
TILE_INDEX_3  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_road_only.csv")
TILE_INDEX_2  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_2class.csv")
FULL_INDEX    = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index.csv")
MODELS_3CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models")
MODELS_2CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models\2class")
TFW_ROOT      = Path(r"\\kgkn-nas\eo_data_2\GURS_podatki\DOF\DOF025\1_DOF025_Geolokacije\DOF025_TFW")
OUT_DIR       = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES  = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']
MODEL_LABELS = {
    'unet_resnet50':  'U-Net/ResNet50',
    'dlinknet':       'D-LinkNet',
    'segformer':      'SegFormer-B2',
    'deeplabv3plus':  'DeepLabV3+'
}

TILE_SIZE      = 512
PIXEL_SIZE     = 0.25
TILE_STEP      = 384
TILE_STEP_M    = TILE_STEP * PIXEL_SIZE
GRID_SIZE      = 3
MIN_PRED_RATIO = 0.10

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       9,
    'axes.titlesize':  10,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight'
})

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

def compute_geo_coords(tile_row):
    tfw = read_tfw(tile_row['source_ortho'])
    if tfw is None:
        return None, None
    x_geo = tfw['x_origin'] + tile_row['x_offset'] * PIXEL_SIZE
    y_geo = tfw['y_origin'] + tile_row['y_offset'] * (-PIXEL_SIZE)
    return x_geo, y_geo

# ── Geographic index ────────────────────────────────────────────────────────
def build_geo_index(tiles_df):
    geo_index = {}
    for _, row in tiles_df.iterrows():
        x_geo, y_geo = compute_geo_coords(row)
        if x_geo is None:
            continue
        key = (round(x_geo, 1), round(y_geo, 1))
        geo_index[key] = row
    return geo_index

def find_geo_grid(center_tile, geo_index, grid_size=3):
    cx, cy = compute_geo_coords(center_tile)
    half   = grid_size // 2
    grid   = []
    for row_idx in range(-half, half + 1):
        grid_row = []
        for col_idx in range(-half, half + 1):
            tx = cx + col_idx * TILE_STEP_M
            ty = cy - row_idx * TILE_STEP_M
            found = None
            for dx in [0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0]:
                for dy in [0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0]:
                    key = (round(tx + dx, 1), round(ty + dy, 1))
                    if key in geo_index:
                        found = geo_index[key]
                        break
                if found is not None:
                    break
            grid_row.append(found)
        grid.append(grid_row)
    found_count = sum(1 for r in grid for t in r if t is not None)
    return grid, found_count

# ── Color maps ─────────────────────────────────────────────────────────────
def mask_to_rgb_3class(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [235, 235, 235]
    rgb[mask == 1] = [210, 40,  40]
    rgb[mask == 2] = [40,  90,  200]
    rgb[mask == 3] = [40,  170, 70]
    return rgb

def mask_to_rgb_2class(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [235, 235, 235]
    rgb[mask == 1] = [210, 40,  40]
    rgb[mask == 2] = [40,  170, 70]
    return rgb

# ── Model loading ──────────────────────────────────────────────────────────
def get_model(model_name, num_classes):
    if model_name == 'unet_resnet50':
        return smp.Unet(encoder_name='resnet50',
                        encoder_weights=None,
                        in_channels=3, classes=num_classes)
    elif model_name == 'dlinknet':
        return smp.Unet(encoder_name='resnet34',
                        encoder_weights=None,
                        in_channels=3, classes=num_classes,
                        decoder_use_batchnorm=True)
    elif model_name == 'segformer':
        return smp.Segformer(encoder_name='mit_b2',
                             encoder_weights=None,
                             in_channels=3, classes=num_classes)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(encoder_name='resnet50',
                                 encoder_weights=None,
                                 in_channels=3, classes=num_classes)

def load_model(model_name, model_dir, num_classes):
    model      = get_model(model_name, num_classes).cuda()
    checkpoint = torch.load(
        model_dir / model_name / f"{model_name}_best.pth",
        map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['epoch']

def predict_single(model, img):
    aug    = test_transform(image=img,
                            mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = aug['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

# ── Canvas extent ──────────────────────────────────────────────────────────
def compute_canvas_extent(grid):
    x_min =  float('inf')
    y_max = -float('inf')
    x_max = -float('inf')
    y_min =  float('inf')
    for row in grid:
        for tile in row:
            if tile is None:
                continue
            x_geo, y_geo = compute_geo_coords(tile)
            if x_geo is None:
                continue
            x_min = min(x_min, x_geo)
            x_max = max(x_max, x_geo + TILE_SIZE * PIXEL_SIZE)
            y_max = max(y_max, y_geo)
            y_min = min(y_min, y_geo - TILE_SIZE * PIXEL_SIZE)
    canvas_w = int(round((x_max - x_min) / PIXEL_SIZE))
    canvas_h = int(round((y_max - y_min) / PIXEL_SIZE))
    return x_min, y_max, canvas_w, canvas_h

# ── Stitching using exact geographic coordinates ────────────────────────────
def stitch_image(grid):
    x_min, y_max, canvas_w, canvas_h = compute_canvas_extent(grid)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 200
    for row in grid:
        for tile in row:
            if tile is None:
                continue
            x_geo, y_geo = compute_geo_coords(tile)
            if x_geo is None:
                continue
            col  = int(round((x_geo - x_min) / PIXEL_SIZE))
            row_ = int(round((y_max - y_geo) / PIXEL_SIZE))
            img  = np.load(tile['image_path'])
            r0 = row_
            r1 = min(row_ + TILE_SIZE, canvas_h)
            c0 = col
            c1 = min(col  + TILE_SIZE, canvas_w)
            canvas[r0:r1, c0:c1] = img[:r1-r0, :c1-c0]
    return canvas

def stitch_mask_3class(grid):
    x_min, y_max, canvas_w, canvas_h = compute_canvas_extent(grid)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for row in grid:
        for tile in row:
            if tile is None:
                continue
            x_geo, y_geo = compute_geo_coords(tile)
            if x_geo is None:
                continue
            col  = int(round((x_geo - x_min) / PIXEL_SIZE))
            row_ = int(round((y_max - y_geo) / PIXEL_SIZE))
            mask = np.load(tile['mask_path'])
            r0 = row_
            r1 = min(row_ + TILE_SIZE, canvas_h)
            c0 = col
            c1 = min(col  + TILE_SIZE, canvas_w)
            canvas[r0:r1, c0:c1] = mask[:r1-r0, :c1-c0]
    return canvas

def stitch_mask_2class(grid_3, grid_2):
    x_min, y_max, canvas_w, canvas_h = compute_canvas_extent(grid_3)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for row_3, row_2 in zip(grid_3, grid_2):
        for tile_3, tile_2 in zip(row_3, row_2):
            if tile_3 is None or tile_2 is None:
                continue
            x_geo, y_geo = compute_geo_coords(tile_3)
            if x_geo is None:
                continue
            col  = int(round((x_geo - x_min) / PIXEL_SIZE))
            row_ = int(round((y_max - y_geo) / PIXEL_SIZE))
            mask = np.load(tile_2['mask_path'])
            r0 = row_
            r1 = min(row_ + TILE_SIZE, canvas_h)
            c0 = col
            c1 = min(col  + TILE_SIZE, canvas_w)
            canvas[r0:r1, c0:c1] = mask[:r1-r0, :c1-c0]
    return canvas

def stitch_prediction(model, grid):
    x_min, y_max, canvas_w, canvas_h = compute_canvas_extent(grid)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for row in grid:
        for tile in row:
            if tile is None:
                continue
            x_geo, y_geo = compute_geo_coords(tile)
            if x_geo is None:
                continue
            col  = int(round((x_geo - x_min) / PIXEL_SIZE))
            row_ = int(round((y_max - y_geo) / PIXEL_SIZE))
            img  = np.load(tile['image_path'])
            pred = predict_single(model, img)
            r0 = row_
            r1 = min(row_ + TILE_SIZE, canvas_h)
            c0 = col
            c1 = min(col  + TILE_SIZE, canvas_w)
            canvas[r0:r1, c0:c1] = pred[:r1-r0, :c1-c0]
    return canvas

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_grid_f1(pred_canvas, mask_canvas, num_classes):
    road_f1 = []
    cls_f1  = {}
    cls_iou = {}
    for cls in range(1, num_classes):
        tp = ((pred_canvas == cls) & (mask_canvas == cls)).sum()
        fp = ((pred_canvas == cls) & (mask_canvas != cls)).sum()
        fn = ((pred_canvas != cls) & (mask_canvas == cls)).sum()
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        iou  = tp / (tp + fp + fn + 1e-6)
        cls_f1[cls]  = float(f1)
        cls_iou[cls] = float(iou)
        road_f1.append(f1)
    return float(np.mean(road_f1)), cls_f1, cls_iou

# ── Find best center tile ──────────────────────────────────────────────────
def find_best_center_tile(muni_tiles, model, num_classes):
    candidates = muni_tiles[
        muni_tiles['road_pixel_ratio'] > 0
    ].sort_values('road_pixel_ratio', ascending=False).head(300)

    best_tile  = None
    best_f1    = -1
    best_ratio = -1

    for _, tile in candidates.iterrows():
        img  = np.load(tile['image_path'])
        pred = predict_single(model, img)

        pred_ratio = (pred > 0).sum() / pred.size
        if pred_ratio < MIN_PRED_RATIO:
            continue

        mask = np.load(tile['mask_path'])
        road_f1, _, _ = compute_grid_f1(pred, mask, num_classes)

        if road_f1 > best_f1:
            best_f1    = road_f1
            best_tile  = tile
            best_ratio = pred_ratio

    if best_tile is None:
        best_tile  = candidates.iloc[0]
        best_f1    = 0
        best_ratio = 0

    return best_tile, best_f1, best_ratio


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("VISUAL MAP GENERATOR — 3×3 SPATIALLY CONTIGUOUS GRIDS")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

tile_df_3 = pd.read_csv(TILE_INDEX_3)
tile_df_2 = pd.read_csv(TILE_INDEX_2)
test_df_3 = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)
test_df_2 = tile_df_2[tile_df_2['split'] == 'test'].reset_index(drop=True)

try:
    full_df   = pd.read_csv(FULL_INDEX)
    test_full = full_df[full_df['split'] == 'test'].reset_index(drop=True)
    print(f"Full test tiles for geo lookup: {len(test_full):,}")
except Exception:
    test_full = test_df_3.copy()
    print("Full index not found — using road-only index")

municipalities = sorted(test_df_3['municipality'].unique())
print(f"Test municipalities: {municipalities}")

# ── Load models ────────────────────────────────────────────────────────────
print("\nLoading 3-class models...")
models_3 = {}
for name in MODEL_NAMES:
    m, ep = load_model(name, MODELS_3CLASS, num_classes=4)
    models_3[name] = m
    print(f"  {name} loaded (epoch {ep})")

print("\nLoading 2-class models...")
models_2 = {}
for name in MODEL_NAMES:
    m, ep = load_model(name, MODELS_2CLASS, num_classes=3)
    models_2[name] = m
    print(f"  {name} loaded (epoch {ep})")

# ── Find best tiles and build grids ───────────────────────────────────────
print("\n=== Finding best center tiles and building 3×3 grids ===")

grids_data = {}

for muni in municipalities:
    print(f"\n--- {muni} ---")
    landscape = test_df_3[
        test_df_3['municipality'] == muni
    ]['landscape_type'].iloc[0].capitalize()

    muni_tiles_3 = test_df_3[test_df_3['municipality'] == muni]
    muni_tiles_2 = test_df_2[test_df_2['municipality'] == muni]
    muni_full    = test_full[test_full['municipality'] == muni]

    print(f"  Finding best center tile...")
    center_tile, best_f1, pred_ratio = find_best_center_tile(
        muni_tiles_3, models_3['segformer'], num_classes=4)
    cx, cy = compute_geo_coords(center_tile)
    print(f"  Best: {center_tile['tile_id']} | "
          f"F1={best_f1:.4f} | pred_ratio={pred_ratio:.3f} | "
          f"geo=({cx:.0f}, {cy:.0f})")

    print(f"  Building geographic index...")
    geo_index_full = build_geo_index(muni_full)

    grid_3, found = find_geo_grid(center_tile, geo_index_full, GRID_SIZE)
    print(f"  Grid tiles found: {found}/{GRID_SIZE*GRID_SIZE}")

    # Build 2-class grid matching same tile_ids
    grid_2 = []
    for row in grid_3:
        grid_row_2 = []
        for tile in row:
            if tile is None:
                grid_row_2.append(None)
            else:
                match = muni_tiles_2[
                    muni_tiles_2['tile_id'] == tile['tile_id']]
                grid_row_2.append(
                    match.iloc[0] if len(match) > 0 else None)
        grid_2.append(grid_row_2)

    # Stitch all layers
    print(f"  Stitching grids with geographic alignment...")
    img_canvas    = stitch_image(grid_3)
    mask_3_canvas = stitch_mask_3class(grid_3)
    mask_2_canvas = stitch_mask_2class(grid_3, grid_2)

    canvas_h, canvas_w = img_canvas.shape[:2]
    coverage_x = canvas_w * PIXEL_SIZE
    coverage_y = canvas_h * PIXEL_SIZE
    print(f"  Canvas: {canvas_w}×{canvas_h} px | "
          f"Coverage: {coverage_x:.0f}×{coverage_y:.0f}m")

    # Predictions — all models, both schemes
    preds_3 = {}
    preds_2 = {}
    f1s_3   = {}
    f1s_2   = {}

    for model_name in MODEL_NAMES:
        print(f"  Predicting {model_name}...", flush=True)
        pred_3 = stitch_prediction(models_3[model_name], grid_3)
        pred_2 = stitch_prediction(models_2[model_name], grid_3)
        preds_3[model_name] = pred_3
        preds_2[model_name] = pred_2
        f1_3, cls_f1_3, cls_iou_3 = compute_grid_f1(
            pred_3, mask_3_canvas, num_classes=4)
        f1_2, cls_f1_2, cls_iou_2 = compute_grid_f1(
            pred_2, mask_2_canvas, num_classes=3)
        f1s_3[model_name] = (f1_3, cls_f1_3, cls_iou_3)
        f1s_2[model_name] = (f1_2, cls_f1_2, cls_iou_2)
        print(f"    3-class F1={f1_3:.4f} | 2-class F1={f1_2:.4f}")

    grids_data[muni] = {
        'landscape':      landscape,
        'center_tile':    center_tile,
        'best_f1':        best_f1,
        'found':          found,
        'coverage_x':     coverage_x,
        'coverage_y':     coverage_y,
        'img_canvas':     img_canvas,
        'mask_3_canvas':  mask_3_canvas,
        'mask_2_canvas':  mask_2_canvas,
        'preds_3':        preds_3,
        'preds_2':        preds_2,
        'f1s_3':          f1s_3,
        'f1s_2':          f1s_2,
    }
    print(f"  Done.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3-CLASS COMBINED
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating Figure 1: 3-class combined ===")

n_rows = len(municipalities)
n_cols = 6

fig = plt.figure(figsize=(28, 6 * n_rows))
gs  = gridspec.GridSpec(n_rows, n_cols,
                         figure=fig,
                         hspace=0.12, wspace=0.03)

col_titles_3 = [
    'Orthophoto', 'Ground Truth',
    'U-Net/ResNet50', 'D-LinkNet',
    'SegFormer-B2', 'DeepLabV3+'
]

for row_idx, muni in enumerate(municipalities):
    d            = grids_data[muni]
    landscape    = d['landscape']
    found        = d['found']
    coverage_x   = d['coverage_x']
    coverage_y   = d['coverage_y']
    best_model_3 = max(d['f1s_3'], key=lambda k: d['f1s_3'][k][0])

    # Col 0 — Orthophoto
    ax = fig.add_subplot(gs[row_idx, 0])
    ax.imshow(d['img_canvas'])
    ax.set_title(
        f"{muni} ({landscape}) | "
        f"~{coverage_x:.0f}×{coverage_y:.0f}m | "
        f"{found}/{GRID_SIZE**2} tiles",
        fontsize=8, fontweight='bold', loc='left', pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_xlabel(col_titles_3[0], fontsize=10,
                      fontweight='bold', labelpad=4)

    # Col 1 — Ground truth
    ax   = fig.add_subplot(gs[row_idx, 1])
    mask = d['mask_3_canvas']
    ax.imshow(mask_to_rgb_3class(mask))
    total = mask.size
    info  = (f"Maj: {(mask==1).sum()/total*100:.1f}%\n"
             f"Loc: {(mask==2).sum()/total*100:.1f}%\n"
             f"Min: {(mask==3).sum()/total*100:.1f}%")
    ax.text(0.02, 0.02, info,
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', alpha=0.85,
                      edgecolor='grey', linewidth=0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_xlabel(col_titles_3[1], fontsize=10,
                      fontweight='bold', labelpad=4)

    # Cols 2-5 — Model predictions
    for col_idx, model_name in enumerate(MODEL_NAMES):
        ax         = fig.add_subplot(gs[row_idx, col_idx + 2])
        f1, cls_f1, _ = d['f1s_3'][model_name]
        is_best    = (model_name == best_model_3)

        ax.imshow(mask_to_rgb_3class(d['preds_3'][model_name]))

        txt = (f"F1:  {f1:.3f}\n"
               f"Maj: {cls_f1[1]:.3f}\n"
               f"Loc: {cls_f1[2]:.3f}\n"
               f"Min: {cls_f1[3]:.3f}")
        fc = '#d4edda' if is_best else 'white'
        ec = '#28a745' if is_best else '#999999'
        lw = 2.0      if is_best else 0.8
        ax.text(0.02, 0.02, txt,
                transform=ax.transAxes,
                fontsize=7, va='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=fc, edgecolor=ec,
                          alpha=0.92, linewidth=lw))

        if is_best:
            ax.text(0.98, 0.98, '★ Best',
                    transform=ax.transAxes,
                    fontsize=8, ha='right', va='top',
                    color='#28a745', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('#28a745')
                spine.set_linewidth(2.5)

        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_xlabel(col_titles_3[col_idx + 2],
                          fontsize=10, fontweight='bold', labelpad=4)

legend_3 = [
    Patch(facecolor='#EBEBEB', edgecolor='grey', label='Background'),
    Patch(facecolor='#D22828', label='Major roads (AC/HC/G1/G2/R1)'),
    Patch(facecolor='#285AC8', label='Local roads (R2/R3/LC/LZ/RT)'),
    Patch(facecolor='#28AA46', label='Minor roads (LG/LK/JP/NK/PP)'),
]
fig.legend(handles=legend_3, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.012),
           frameon=True, edgecolor='grey', fancybox=True)

fig.suptitle(
    'Functional Road Extraction — 3-Class Segmentation Results\n'
    'Test Municipalities | 3×3 Spatially Contiguous Grid (TFW-aligned) | '
    '★ = Best model per municipality',
    fontsize=12, fontweight='bold', y=1.005)

plt.savefig(OUT_DIR / 'fig_predictions_3class_combined.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_predictions_3class_combined.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 2-CLASS COMBINED
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating Figure 2: 2-class combined ===")

fig = plt.figure(figsize=(28, 6 * n_rows))
gs  = gridspec.GridSpec(n_rows, n_cols,
                         figure=fig,
                         hspace=0.12, wspace=0.03)

col_titles_2 = [
    'Orthophoto', 'Ground Truth',
    'U-Net/ResNet50', 'D-LinkNet',
    'SegFormer-B2', 'DeepLabV3+'
]

for row_idx, muni in enumerate(municipalities):
    d            = grids_data[muni]
    landscape    = d['landscape']
    found        = d['found']
    coverage_x   = d['coverage_x']
    coverage_y   = d['coverage_y']
    best_model_2 = max(d['f1s_2'], key=lambda k: d['f1s_2'][k][0])

    # Col 0 — Orthophoto
    ax = fig.add_subplot(gs[row_idx, 0])
    ax.imshow(d['img_canvas'])
    ax.set_title(
        f"{muni} ({landscape}) | "
        f"~{coverage_x:.0f}×{coverage_y:.0f}m | "
        f"{found}/{GRID_SIZE**2} tiles",
        fontsize=8, fontweight='bold', loc='left', pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_xlabel(col_titles_2[0], fontsize=10,
                      fontweight='bold', labelpad=4)

    # Col 1 — GT 2-class
    ax   = fig.add_subplot(gs[row_idx, 1])
    mask = d['mask_2_canvas']
    ax.imshow(mask_to_rgb_2class(mask))
    total = mask.size
    info  = (f"Pri: {(mask==1).sum()/total*100:.1f}%\n"
             f"Sec: {(mask==2).sum()/total*100:.1f}%")
    ax.text(0.02, 0.02, info,
            transform=ax.transAxes, fontsize=7, va='bottom',
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', alpha=0.85,
                      edgecolor='grey', linewidth=0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_xlabel(col_titles_2[1], fontsize=10,
                      fontweight='bold', labelpad=4)

    # Cols 2-5 — Model predictions
    for col_idx, model_name in enumerate(MODEL_NAMES):
        ax         = fig.add_subplot(gs[row_idx, col_idx + 2])
        f1, cls_f1, _ = d['f1s_2'][model_name]
        is_best    = (model_name == best_model_2)

        ax.imshow(mask_to_rgb_2class(d['preds_2'][model_name]))

        txt = (f"F1:  {f1:.3f}\n"
               f"Pri: {cls_f1[1]:.3f}\n"
               f"Sec: {cls_f1[2]:.3f}")
        fc = '#d4edda' if is_best else 'white'
        ec = '#28a745' if is_best else '#999999'
        lw = 2.0      if is_best else 0.8
        ax.text(0.02, 0.02, txt,
                transform=ax.transAxes,
                fontsize=7, va='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=fc, edgecolor=ec,
                          alpha=0.92, linewidth=lw))

        if is_best:
            ax.text(0.98, 0.98, '★ Best',
                    transform=ax.transAxes,
                    fontsize=8, ha='right', va='top',
                    color='#28a745', fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('#28a745')
                spine.set_linewidth(2.5)

        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_xlabel(
                col_titles_2[col_idx + 2],
                fontsize=10, fontweight='bold', labelpad=4)

legend_2 = [
    Patch(facecolor='#EBEBEB', edgecolor='grey', label='Background'),
    Patch(facecolor='#D22828', label='Primary roads (Major + Local)'),
    Patch(facecolor='#28AA46', label='Secondary roads (Minor/unclassified)'),
]
fig.legend(handles=legend_2, loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.012),
           frameon=True, edgecolor='grey', fancybox=True)

fig.suptitle(
    'Functional Road Extraction — 2-Class Segmentation Results\n'
    'Test Municipalities | 3×3 Spatially Contiguous Grid (TFW-aligned) | '
    '★ = Best model per municipality',
    fontsize=12, fontweight='bold', y=1.005)

plt.savefig(OUT_DIR / 'fig_predictions_2class_combined.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_predictions_2class_combined.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 3-CLASS vs 2-CLASS COMPARISON (SegFormer only)
# ══════════════════════════════════════════════════════════════════════════
print("\n=== Generating Figure 3: 3-class vs 2-class comparison ===")

fig = plt.figure(figsize=(23, 6 * n_rows))
gs  = gridspec.GridSpec(n_rows, 5,
                         figure=fig,
                         hspace=0.12, wspace=0.03)

col_titles_cmp = [
    'Orthophoto',
    'Ground Truth — 3-class',
    'SegFormer-B2 — 3-class',
    'Ground Truth — 2-class',
    'SegFormer-B2 — 2-class',
]

for row_idx, muni in enumerate(municipalities):
    d         = grids_data[muni]
    landscape = d['landscape']
    coverage_x = d['coverage_x']
    coverage_y = d['coverage_y']
    f1_3      = d['f1s_3']['segformer'][0]
    f1_2      = d['f1s_2']['segformer'][0]
    delta     = f1_2 - f1_3
    delta_col = '#28a745' if delta >= 0 else '#dc3545'

    panels = [
        (d['img_canvas'],                               None,                       'black',   0),
        (mask_to_rgb_3class(d['mask_3_canvas']),        None,                       'black',   1),
        (mask_to_rgb_3class(d['preds_3']['segformer']), f"Road F1: {f1_3:.3f}",     'black',   2),
        (mask_to_rgb_2class(d['mask_2_canvas']),        None,                       'black',   3),
        (mask_to_rgb_2class(d['preds_2']['segformer']),
         f"Road F1: {f1_2:.3f}\nΔ: {delta:+.3f}",     delta_col,  4),
    ]

    for canvas, label_txt, label_color, col_idx in panels:
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.imshow(canvas)

        if col_idx == 0:
            ax.set_title(
                f"{muni} ({landscape}) | "
                f"~{coverage_x:.0f}×{coverage_y:.0f}m",
                fontsize=8, fontweight='bold',
                loc='left', pad=3)

        if label_txt:
            ax.text(0.02, 0.02, label_txt,
                    transform=ax.transAxes,
                    fontsize=8, va='bottom',
                    color=label_color,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', alpha=0.88,
                              edgecolor='grey', linewidth=0.8))

        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_xlabel(col_titles_cmp[col_idx],
                          fontsize=10, fontweight='bold', labelpad=4)

legend_cmp = [
    Patch(facecolor='#EBEBEB', edgecolor='grey', label='Background'),
    Patch(facecolor='#D22828', label='Major / Primary roads'),
    Patch(facecolor='#285AC8', label='Local roads (3-class only)'),
    Patch(facecolor='#28AA46', label='Minor / Secondary roads'),
]
fig.legend(handles=legend_cmp, loc='lower center', ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.012),
           frameon=True, edgecolor='grey', fancybox=True)

fig.suptitle(
    'SegFormer-B2: 3-Class vs 2-Class Prediction Comparison\n'
    'Test Municipalities | 3×3 Spatially Contiguous Grid (TFW-aligned) | '
    'Δ = F1 improvement of 2-class over 3-class',
    fontsize=12, fontweight='bold', y=1.005)

plt.savefig(OUT_DIR / 'fig_3class_vs_2class_visual_comparison.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_3class_vs_2class_visual_comparison.png")


print(f"\n{'='*70}")
print(f"ALL VISUAL MAPS SAVED TO: {OUT_DIR}")
print(f"{'='*70}")
for f in sorted(OUT_DIR.glob('fig_pred*.png')):
    print(f"  {f.name}")
for f in sorted(OUT_DIR.glob('fig_3class_vs*.png')):
    print(f"  {f.name}")