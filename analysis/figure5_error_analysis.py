"""
figure5_error_analysis.py
Builds Figure 5 — Error Analysis for ISPRS submission.
Shows four cases with annotated confusion zones:
  Row 1 — Urban FP:      rooftop/car-park false positives (Nova Gorica)
  Row 2 — Coastal FP:    hard-surface false positives (Piran)
  Row 3 — Alpine FN:     canopy/forest omissions (Bohinj)
  Row 4 — Success:       clean agricultural detection (Lendava)

Each row: Orthophoto | Ground Truth | Prediction | Difference map
Annotations mark the specific confusion zones.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
TILE_INDEX_3  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_road_only.csv")
COUNTS_CSV    = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article\artifact1_per_tile_counts.csv")
MODELS_3CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models")
OUT_DIR       = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Font sizes ─────────────────────────────────────────────────────────────
FS_COL_TITLE  = 11
FS_ROW_LABEL  = 10
FS_ANNOTATION = 8
FS_LEGEND     = 10
FS_SUPTITLE   = 13
FS_METRICS    = 8

# ── Transform ──────────────────────────────────────────────────────────────
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── Models ─────────────────────────────────────────────────────────────────
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

# ── Color maps ─────────────────────────────────────────────────────────────
def mask_to_rgb(mask):
    """3-class road mask to RGB."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [235, 235, 235]   # background — light grey
    rgb[mask == 1] = [210, 40,  40]    # major — red
    rgb[mask == 2] = [40,  90,  200]   # local — blue
    rgb[mask == 3] = [40,  170, 70]    # minor — green
    return rgb

def difference_map(pred, gt):
    """
    Pixel-level error map:
      True Positive  (road correct)  → green
      False Positive (pred road, no gt road) → red
      False Negative (gt road, no pred road) → orange
      True Negative  (background correct)    → light grey
    """
    rgb = np.ones((*pred.shape, 3), dtype=np.uint8) * 235
    pred_road = (pred > 0)
    gt_road   = (gt   > 0)
    # TP — correct road detection
    tp = pred_road &  gt_road
    # FP — false positive road
    fp = pred_road & ~gt_road
    # FN — missed road
    fn = ~pred_road & gt_road

    rgb[tp] = [40,  190, 70]    # green — correct
    rgb[fp] = [220, 50,  50]    # red   — false positive
    rgb[fn] = [255, 165, 0]     # orange — false negative
    return rgb, tp, fp, fn

# ── Metrics ────────────────────────────────────────────────────────────────
def tile_metrics(pred, gt):
    road_f1 = []
    for cls in range(1, 4):
        tp = ((pred==cls) & (gt==cls)).sum()
        fp = ((pred==cls) & (gt!=cls)).sum()
        fn = ((pred!=cls) & (gt==cls)).sum()
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        road_f1.append(float(f1))
    return float(np.mean(road_f1))

def compute_error_rates(pred, gt):
    pred_road = (pred > 0)
    gt_road   = (gt   > 0)
    total_gt  = gt_road.sum()
    total_pred = pred_road.sum()
    fp = (pred_road & ~gt_road).sum()
    fn = (~pred_road & gt_road).sum()
    fp_rate = float(fp / (total_pred + 1e-6))  # FP as % of predicted
    fn_rate = float(fn / (total_gt   + 1e-6))  # FN as % of reference
    return fp_rate, fn_rate

# ── Annotation helper ──────────────────────────────────────────────────────
def add_annotation_box(ax, x, y, w, h,
                        label, color,
                        text_x=None, text_y=None,
                        arrow_dx=0, arrow_dy=0):
    """
    Draw a rectangle around a confusion zone with an arrow label.
    x, y, w, h in pixel coordinates (0–512).
    """
    # Convert pixel coords to axes fraction
    img_h, img_w = 512, 512
    ax_x  = x / img_w
    ax_y  = 1.0 - (y + h) / img_h   # flip y
    ax_w  = w / img_w
    ax_h  = h / img_h

    rect = Rectangle((ax_x, ax_y), ax_w, ax_h,
                      linewidth=1.8, edgecolor=color,
                      facecolor='none',
                      transform=ax.transAxes, zorder=5)
    ax.add_patch(rect)

    if label:
        tx = text_x if text_x is not None else ax_x + ax_w / 2
        ty = text_y if text_y is not None else ax_y - 0.04

        # Arrow from text to box
        cx  = ax_x + ax_w / 2
        cy  = ax_y + ax_h / 2

        ax.annotate(
            label,
            xy=(cx + arrow_dx, cy + arrow_dy),
            xytext=(tx, ty),
            xycoords='axes fraction',
            textcoords='axes fraction',
            fontsize=FS_ANNOTATION,
            fontweight='bold',
            color='white',
            ha='center', va='center',
            arrowprops=dict(
                arrowstyle='->', color=color,
                lw=1.5,
                connectionstyle='arc3,rad=0.0'),
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5),
            zorder=6)

def add_simple_label(ax, x, y, label, color, fontsize=None):
    """Add a simple text label with colored background."""
    fs = fontsize or FS_ANNOTATION
    ax.text(x, y, label,
            transform=ax.transAxes,
            fontsize=fs, fontweight='bold',
            color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor=color, alpha=0.88,
                      edgecolor='white', linewidth=0.5),
            zorder=6,
            path_effects=[pe.withStroke(linewidth=0.5,
                                         foreground='black')])


# ══════════════════════════════════════════════════════════════════════════
# CASE DEFINITIONS
# Each case: tile_id, municipality, row label, annotation instructions
# Tile IDs confirmed by Artifact 3
# ══════════════════════════════════════════════════════════════════════════
CASES = [
    {
        'tile_id':    'B052458C_y09984_x04992',
        'muni':       'Nova Gorica',
        'landscape':  'Urban',
        'row_label':  'Urban (Nova Gorica)\nRooftop/car-park confusion',
        'fp_label':   'FP: rooftop\nconfused\nwith road',
        'fn_label':   'FN: road\nomitted',
        'fp_color':   '#DC3232',
        'fn_color':   '#FF8C00',
        'caption':    ('Urban: rooftop and car-park surfaces share the '
                       'spectral signature of asphalt at 0.25 m GSD, '
                       'producing systematic false positives.'),
    },
    {
        'tile_id':    'B011458D_y03840_x02688',
        'muni':       'Piran',
        'landscape':  'Coastal',
        'row_label':  'Coastal (Piran)\nHard-surface confusion',
        'fp_label':   'FP: coastal\nhard surface',
        'fn_label':   'FN: road\nomitted',
        'fp_color':   '#DC3232',
        'fn_color':   '#FF8C00',
        'caption':    ('Coastal: paved promenades, harbour quays, and '
                       'parking areas adjacent to the sea are '
                       'misclassified as roads.'),
    },
    {
        'tile_id':    'C083458C_y05760_x01152',
        'muni':       'Bohinj',
        'landscape':  'Alpine',
        'row_label':  'Alpine (Bohinj)\nCanopy / shadow omission',
        'fp_label':   'FP: shadow\nor bare soil',
        'fn_label':   'FN: road\nunder canopy',
        'fp_color':   '#DC3232',
        'fn_color':   '#FF8C00',
        'caption':    ('Alpine: tree canopy and cast shadows obscure '
                       'the road surface, causing false negatives; '
                       'bare soil at forest edges causes false positives.'),
    },
    {
        'tile_id':    'K105058E_y02304_x05376',
        'muni':       'Lendava',
        'landscape':  'Agricultural',
        'row_label':  'Agricultural (Lendava)\nRepresentative success case',
        'fp_label':   '',
        'fn_label':   '',
        'fp_color':   '#DC3232',
        'fn_color':   '#FF8C00',
        'caption':    ('Agricultural: flat open terrain with unobstructed '
                       'road surfaces yields near-perfect detection '
                       '(Road Macro F1 = 0.928).'),
    },
]


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("FIGURE 5 — ERROR ANALYSIS")
print("="*70)

tile_df_3 = pd.read_csv(TILE_INDEX_3)
test_df_3 = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)

print("\nLoading SegFormer-B2...")
segformer = load_segformer()
print("Loaded.")

# ── Load all tile data ─────────────────────────────────────────────────────
print("\nLoading tiles and running predictions...")
tile_data = []
for case in CASES:
    tile_row = test_df_3[
        test_df_3['tile_id'] == case['tile_id']
    ].iloc[0]

    img  = np.load(tile_row['image_path'])
    gt   = np.load(tile_row['mask_path'])
    pred = predict(segformer, img)
    diff, tp, fp, fn = difference_map(pred, gt)
    f1   = tile_metrics(pred, gt)
    fp_rate, fn_rate = compute_error_rates(pred, gt)

    tile_data.append({
        **case,
        'img':     img,
        'gt':      gt,
        'pred':    pred,
        'diff':    diff,
        'tp':      tp,
        'fp':      fp,
        'fn':      fn,
        'f1':      f1,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
    })
    print(f"  {case['muni']:15s} | F1={f1:.3f} | "
          f"FP rate={fp_rate*100:.1f}% | FN rate={fn_rate*100:.1f}%")

# ── Build figure ───────────────────────────────────────────────────────────
print("\nBuilding Figure 5...")

n_rows = len(CASES)
n_cols = 4

# Row height: image rows taller, caption rows short
row_ratios = []
for _ in range(n_rows):
    row_ratios.append(8)    # image row
    row_ratios.append(1)    # caption row

fig = plt.figure(figsize=(20, 5.5 * n_rows))

# Use gridspec with extra caption rows
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(
    n_rows * 2, n_cols,
    figure=fig,
    height_ratios=row_ratios,
    hspace=0.04,
    wspace=0.04,
    top=0.95, bottom=0.06,
    left=0.01, right=0.99)

col_titles = [
    'Orthophoto',
    'Ground Truth (3-class)',
    'SegFormer-B2 Prediction',
    'Error Map'
]

for row_idx, d in enumerate(tile_data):
    img_row     = row_idx * 2
    caption_row = row_idx * 2 + 1

    gt_rgb   = mask_to_rgb(d['gt'])
    pred_rgb = mask_to_rgb(d['pred'])

    # ── Col 0 — Orthophoto ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[img_row, 0])
    ax.imshow(d['img'])
    ax.set_xticks([])
    ax.set_yticks([])

    # Row label on left
    ax.set_ylabel(d['row_label'],
                  fontsize=FS_ROW_LABEL,
                  fontweight='bold',
                  rotation=90, labelpad=6, va='center')

    # Column titles on first row only
    if row_idx == 0:
        ax.set_title(col_titles[0], fontsize=FS_COL_TITLE,
                     fontweight='bold', pad=5)

    # Metrics badge bottom-right
    total   = d['gt'].size
    road_gt = (d['gt'] > 0).sum()
    add_simple_label(ax, 0.97, 0.04,
                     f"Road: {road_gt/total*100:.1f}%",
                     '#444444', fontsize=FS_METRICS - 1)

    # ── Col 1 — Ground Truth ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[img_row, 1])
    ax.imshow(gt_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_title(col_titles[1], fontsize=FS_COL_TITLE,
                     fontweight='bold', pad=5)

    total = d['gt'].size
    info  = (f"Maj: {(d['gt']==1).sum()/total*100:.1f}%  "
             f"Loc: {(d['gt']==2).sum()/total*100:.1f}%  "
             f"Min: {(d['gt']==3).sum()/total*100:.1f}%")
    ax.set_xlabel(info, fontsize=FS_METRICS, labelpad=3)

    # ── Col 2 — Prediction ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[img_row, 2])
    ax.imshow(pred_rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_title(col_titles[2], fontsize=FS_COL_TITLE,
                     fontweight='bold', pad=5)

    # F1 badge
    add_simple_label(ax, 0.5, 0.96,
                     f"Road F1 = {d['f1']:.3f}",
                     '#1a6b1a' if d['f1'] > 0.6 else '#8b1a1a',
                     fontsize=FS_METRICS)

    # ── Col 3 — Error / Difference map ────────────────────────────────────
    ax = fig.add_subplot(gs[img_row, 3])
    ax.imshow(d['diff'])
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_title(col_titles[3], fontsize=FS_COL_TITLE,
                     fontweight='bold', pad=5)

    # FP and FN rate badges
    add_simple_label(ax, 0.28, 0.96,
                     f"FP: {d['fp_rate']*100:.1f}%",
                     '#DC3232', fontsize=FS_METRICS)
    add_simple_label(ax, 0.72, 0.96,
                     f"FN: {d['fn_rate']*100:.1f}%",
                     '#FF8C00', fontsize=FS_METRICS)

    # ── Automatic annotation boxes ────────────────────────────────────────
    # Find largest FP region centroid for annotation placement
    if d['fp'].sum() > 100:
        fp_arr = d['fp'].astype(np.uint8)
        # Find the largest connected blob of FP pixels
        from scipy import ndimage
        labeled, n_features = ndimage.label(fp_arr)
        if n_features > 0:
            sizes    = ndimage.sum(fp_arr, labeled,
                                   range(1, n_features + 1))
            largest  = int(np.argmax(sizes)) + 1
            blob     = (labeled == largest)
            rows_idx = np.where(blob.any(axis=1))[0]
            cols_idx = np.where(blob.any(axis=0))[0]
            if len(rows_idx) > 0 and len(cols_idx) > 0:
                ry0, ry1 = rows_idx[0],  rows_idx[-1]
                cx0, cx1 = cols_idx[0],  cols_idx[-1]
                margin   = 8
                # Draw on prediction column (col 2)
                ax_pred = fig.axes[row_idx * 4 + 2]  # col 2
                add_annotation_box(
                    ax_pred,
                    x=max(0, cx0 - margin),
                    y=max(0, ry0 - margin),
                    w=min(512, cx1 - cx0 + 2*margin),
                    h=min(512, ry1 - ry0 + 2*margin),
                    label=d['fp_label'] if d['fp_label'] else None,
                    color=d['fp_color'],
                    text_x=0.5,
                    text_y=0.06)

    # Find largest FN region for annotation
    if d['fn'].sum() > 100 and d['fn_label']:
        fn_arr = d['fn'].astype(np.uint8)
        from scipy import ndimage
        labeled, n_features = ndimage.label(fn_arr)
        if n_features > 0:
            sizes   = ndimage.sum(fn_arr, labeled,
                                  range(1, n_features + 1))
            largest = int(np.argmax(sizes)) + 1
            blob    = (labeled == largest)
            rows_idx = np.where(blob.any(axis=1))[0]
            cols_idx = np.where(blob.any(axis=0))[0]
            if len(rows_idx) > 0 and len(cols_idx) > 0:
                ry0, ry1 = rows_idx[0],  rows_idx[-1]
                cx0, cx1 = cols_idx[0],  cols_idx[-1]
                margin   = 8
                ax_pred  = fig.axes[row_idx * 4 + 2]
                add_annotation_box(
                    ax_pred,
                    x=max(0, cx0 - margin),
                    y=max(0, ry0 - margin),
                    w=min(512, cx1 - cx0 + 2*margin),
                    h=min(512, ry1 - ry0 + 2*margin),
                    label=d['fn_label'],
                    color=d['fn_color'],
                    text_x=0.5,
                    text_y=0.94)

    # ── Caption row ────────────────────────────────────────────────────────
    ax_cap = fig.add_subplot(gs[caption_row, :])
    ax_cap.axis('off')
    ax_cap.text(
        0.5, 0.5,
        f"({chr(97 + row_idx)}) {d['caption']}",
        transform=ax_cap.transAxes,
        fontsize=FS_ANNOTATION + 0.5,
        style='italic',
        ha='center', va='center',
        color='#222222')

# ── Legend ─────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor='#EBEBEB', edgecolor='grey',
                   label='Background (TN)'),
    mpatches.Patch(facecolor='#28BE46',
                   label='Correct road detection (TP)'),
    mpatches.Patch(facecolor='#DC3232',
                   label='False positive — predicted road, no reference'),
    mpatches.Patch(facecolor='#FFA500',
                   label='False negative — missed road'),
    mpatches.Patch(facecolor='#D22828',
                   label='Major roads (reference)'),
    mpatches.Patch(facecolor='#285AC8',
                   label='Local roads (reference)'),
    mpatches.Patch(facecolor='#28AA46',
                   label='Minor roads (reference)'),
]

fig.legend(handles=legend_handles,
           loc='lower center',
           ncol=4,
           fontsize=FS_LEGEND,
           bbox_to_anchor=(0.5, 0.0),
           frameon=True,
           edgecolor='grey',
           fancybox=True,
           borderpad=0.5,
           handlelength=1.2)

fig.suptitle(
    'Figure 5. Error Analysis — Representative Confusion Cases\n'
    'SegFormer-B2 | 3-Class Scheme | '
    'Red boxes = false positive zones | Orange boxes = false negative zones',
    fontsize=FS_SUPTITLE, fontweight='bold',
    y=0.99)

out_path = OUT_DIR / 'fig5_error_analysis.png'
plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSaved: {out_path}")

# ── Print summary for paper writing ───────────────────────────────────────
print(f"\n{'='*70}")
print("NUMBERS FOR SECTION 5.4")
print(f"{'='*70}")
for d in tile_data:
    print(f"\n{d['muni']} ({d['landscape']}):")
    print(f"  Tile:        {d['tile_id']}")
    print(f"  Road F1:     {d['f1']:.3f}")
    print(f"  FP rate:     {d['fp_rate']*100:.1f}% of predicted pixels")
    print(f"  FN rate:     {d['fn_rate']*100:.1f}% of reference pixels")
    fp_px = d['fp'].sum()
    fn_px = d['fn'].sum()
    tp_px = d['tp'].sum()
    total = d['gt'].size
    print(f"  TP pixels:   {tp_px:,}  ({tp_px/total*100:.1f}%)")
    print(f"  FP pixels:   {fp_px:,}  ({fp_px/total*100:.1f}%)")
    print(f"  FN pixels:   {fn_px:,}  ({fn_px/total*100:.1f}%)")