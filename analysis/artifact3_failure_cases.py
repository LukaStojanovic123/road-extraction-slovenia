"""
artifact3_failure_cases.py
Exports failure-case and success-case crops for error analysis figure.
Uses artifact1_per_tile_counts.csv to select tiles objectively.
Failure types:
  - Urban rooftop/car-park FP  (Nova Gorica)
  - Coastal hard-surface FP    (Piran)
  - Canopy/forest omission FN  (Bohinj)
  - Agricultural success       (Lendava — high F1 contrast)
Output: outputs/article/artifact3_failure_cases/
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths ──────────────────────────────────────────────────────────────────
TILE_INDEX_3  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_road_only.csv")
COUNTS_CSV    = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article\artifact1_per_tile_counts.csv")
MODELS_3CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models")
OUT_DIR       = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article\artifact3_failure_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cases to extract ───────────────────────────────────────────────────────
# Each entry: (label, municipality, failure_type, sort_col, ascending)
# failure_type options:
#   'fp_heavy'  — high false positives → sort by fp_major + fp_local desc
#   'fn_heavy'  — high false negatives → sort by fn_minor desc
#   'success'   — high F1             → sort by road_f1 desc
CASES = [
    ('urban_fp',      'Nova Gorica', 'fp_heavy', None),
    ('coastal_fp',    'Piran',       'fp_heavy', None),
    ('alpine_fn',     'Bohinj',      'fn_heavy', None),
    ('suburban_mixed','Domžale',     'mixed',    None),
    ('agricultural_success', 'Lendava', 'success', None),
]

CROP_SIZE = 512  # pixels — one full tile

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

def load_unet():
    model = smp.Unet(encoder_name='resnet50',
                     encoder_weights=None,
                     in_channels=3, classes=4).cuda()
    ckpt  = torch.load(
        MODELS_3CLASS / 'unet_resnet50' / 'unet_resnet50_best.pth',
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
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [235, 235, 235]
    rgb[mask == 1] = [210, 40,  40]
    rgb[mask == 2] = [40,  90,  200]
    rgb[mask == 3] = [40,  170, 70]
    return rgb

# ── Tile selector ──────────────────────────────────────────────────────────
def compute_tile_f1(row):
    f1s = []
    for cls in ['major', 'local', 'minor']:
        tp = row.get(f'tp_{cls}', 0)
        fp = row.get(f'fp_{cls}', 0)
        fn = row.get(f'fn_{cls}', 0)
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        f1s.append(f1)
    return float(np.mean(f1s))

def select_tile(counts_df, tile_df, municipality,
                failure_type, model='segformer',
                min_road_pixels=0.02):
    """Select representative tile for a given failure type."""
    mdf = counts_df[
        (counts_df['municipality'] == municipality) &
        (counts_df['model']        == model) &
        (counts_df['scheme']       == '3class')
    ].copy()

    # Compute road F1 per tile
    mdf['road_f1'] = mdf.apply(compute_tile_f1, axis=1)

    # Filter tiles with enough road content
    tile_meta = tile_df[tile_df['municipality'] == municipality]
    road_tiles = tile_meta[
        tile_meta['road_pixel_ratio'] >= min_road_pixels
    ]['tile_id'].tolist()
    mdf = mdf[mdf['tile_id'].isin(road_tiles)]

    if failure_type == 'fp_heavy':
        # High false positives — model predicts road where none exists
        mdf['fp_total'] = (mdf.get('fp_major', 0) +
                           mdf.get('fp_local', 0) +
                           mdf.get('fp_minor', 0))
        mdf = mdf[mdf['road_f1'] < 0.5]  # genuinely poor tile
        tile_id = mdf.nlargest(1, 'fp_total')['tile_id'].values[0]

    elif failure_type == 'fn_heavy':
        # High false negatives — model misses roads
        mdf['fn_total'] = (mdf.get('fn_major', 0) +
                           mdf.get('fn_local', 0) +
                           mdf.get('fn_minor', 0))
        mdf = mdf[mdf['road_f1'] < 0.5]
        tile_id = mdf.nlargest(1, 'fn_total')['tile_id'].values[0]

    elif failure_type == 'success':
        # High F1 — clean detection
        tile_id = mdf.nlargest(1, 'road_f1')['tile_id'].values[0]

    elif failure_type == 'mixed':
        # Median performer — representative
        median_f1 = mdf['road_f1'].median()
        tile_id   = mdf.iloc[
            (mdf['road_f1'] - median_f1).abs().argsort()[:1]
        ]['tile_id'].values[0]

    tile_row = tile_meta[tile_meta['tile_id'] == tile_id].iloc[0]
    tile_f1  = mdf[mdf['tile_id'] == tile_id]['road_f1'].values[0]
    return tile_row, tile_f1


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("ARTIFACT 3 — FAILURE CASE CROPS")
print("="*70)

# Load data
tile_df_3  = pd.read_csv(TILE_INDEX_3)
test_df_3  = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)

if not COUNTS_CSV.exists():
    print("ERROR: artifact1_per_tile_counts.csv not found.")
    print("Run artifact1_per_tile_counts.py first.")
    exit(1)

counts_df = pd.read_csv(COUNTS_CSV)
print(f"Loaded {len(counts_df):,} tile count records")

# Load models
print("\nLoading SegFormer and U-Net...")
segformer = load_segformer()
unet      = load_unet()
print("Models loaded.")

# ── Generate figure for each case ─────────────────────────────────────────
case_records = []

for label, municipality, failure_type, _ in CASES:
    print(f"\n--- {label} ({municipality}, {failure_type}) ---")

    tile_row, tile_f1 = select_tile(
        counts_df, test_df_3,
        municipality=municipality,
        failure_type=failure_type)

    print(f"  Selected tile: {tile_row['tile_id']} | F1={tile_f1:.4f}")
    print(f"  Road ratio: {tile_row['road_pixel_ratio']:.3f}")

    img        = np.load(tile_row['image_path'])
    mask       = np.load(tile_row['mask_path'])
    pred_seg   = predict(segformer, img)
    pred_unet  = predict(unet, img)

    # Compute per-model metrics for this tile
    def tile_f1_model(pred, mask):
        f1s = []
        for cls in range(1, 4):
            tp = ((pred==cls) & (mask==cls)).sum()
            fp = ((pred==cls) & (mask!=cls)).sum()
            fn = ((pred!=cls) & (mask==cls)).sum()
            p  = tp / (tp + fp + 1e-6)
            r  = tp / (tp + fn + 1e-6)
            f1 = 2 * p * r / (p + r + 1e-6)
            f1s.append(float(f1))
        return float(np.mean(f1s))

    f1_seg  = tile_f1_model(pred_seg, mask)
    f1_unet = tile_f1_model(pred_unet, mask)

    # ── Individual PNG crops ───────────────────────────────────────────────
    for name, data in [
        ('ortho', img),
        ('gt',    mask_to_rgb(mask)),
        ('segformer', mask_to_rgb(pred_seg)),
        ('unet',  mask_to_rgb(pred_unet))
    ]:
        fname = OUT_DIR / f"{label}_{name}.png"
        plt.imsave(str(fname), data)

    # ── Combined panel figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].set_title('Orthophoto', fontweight='bold', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(mask_to_rgb(mask))
    total  = mask.size
    info   = (f"Maj: {(mask==1).sum()/total*100:.1f}%\n"
              f"Loc: {(mask==2).sum()/total*100:.1f}%\n"
              f"Min: {(mask==3).sum()/total*100:.1f}%")
    axes[1].text(0.02, 0.02, info, transform=axes[1].transAxes,
                 fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes[1].set_title('Ground Truth', fontweight='bold', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(mask_to_rgb(pred_seg))
    axes[2].text(0.02, 0.02, f"Road F1: {f1_seg:.3f}",
                 transform=axes[2].transAxes, fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes[2].set_title('SegFormer-B2', fontweight='bold', fontsize=11)
    axes[2].axis('off')

    axes[3].imshow(mask_to_rgb(pred_unet))
    axes[3].text(0.02, 0.02, f"Road F1: {f1_unet:.3f}",
                 transform=axes[3].transAxes, fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    axes[3].set_title('U-Net/ResNet50', fontweight='bold', fontsize=11)
    axes[3].axis('off')

    legend = [
        Patch(facecolor='#EBEBEB', edgecolor='grey', label='Background'),
        Patch(facecolor='#D22828', label='Major roads'),
        Patch(facecolor='#285AC8', label='Local roads'),
        Patch(facecolor='#28AA46', label='Minor roads'),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.02),
               frameon=True, edgecolor='grey')

    landscape = tile_row['landscape_type'].capitalize()
    fig.suptitle(
        f"{label.replace('_', ' ').title()} — {municipality} ({landscape})\n"
        f"Tile: {tile_row['tile_id']} | "
        f"Road ratio: {tile_row['road_pixel_ratio']*100:.1f}% | "
        f"Failure type: {failure_type}",
        fontsize=12, fontweight='bold')

    plt.tight_layout()
    panel_path = OUT_DIR / f"{label}_panel.png"
    plt.savefig(str(panel_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved panel: {panel_path.name}")

    case_records.append({
        'label':        label,
        'municipality': municipality,
        'landscape':    tile_row['landscape_type'],
        'failure_type': failure_type,
        'tile_id':      tile_row['tile_id'],
        'road_ratio':   tile_row['road_pixel_ratio'],
        'f1_segformer': f1_seg,
        'f1_unet':      f1_unet,
    })

# ── Save case summary ──────────────────────────────────────────────────────
summary_df = pd.DataFrame(case_records)
summary_df.to_csv(OUT_DIR / 'artifact3_case_summary.csv', index=False)

# ── Combined figure — all cases in one ────────────────────────────────────
print("\n  Generating combined all-cases figure...")
n_cases = len(CASES)
fig, axes = plt.subplots(n_cases, 4,
                          figsize=(20, 5 * n_cases))

for row_idx, rec in enumerate(case_records):
    tile_meta = test_df_3[
        test_df_3['tile_id'] == rec['tile_id']].iloc[0]
    img       = np.load(tile_meta['image_path'])
    mask      = np.load(tile_meta['mask_path'])
    pred_seg  = predict(segformer, img)
    pred_unet = predict(unet, img)

    row_title = (f"{rec['label'].replace('_',' ').title()} — "
                 f"{rec['municipality']} ({rec['landscape'].capitalize()})")

    for col_idx, (title, data) in enumerate([
        ('Orthophoto',      img),
        ('Ground Truth',    mask_to_rgb(mask)),
        ('SegFormer-B2',    mask_to_rgb(pred_seg)),
        ('U-Net/ResNet50',  mask_to_rgb(pred_unet)),
    ]):
        ax = axes[row_idx, col_idx]
        ax.imshow(data)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        if col_idx == 0:
            ax.set_xlabel(row_title, fontsize=9,
                          fontweight='bold', labelpad=4)
        if col_idx in [2, 3]:
            f1 = rec['f1_segformer'] if col_idx == 2 else rec['f1_unet']
            ax.text(0.02, 0.02, f"F1: {f1:.3f}",
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.85))

legend = [
    Patch(facecolor='#EBEBEB', edgecolor='grey', label='Background'),
    Patch(facecolor='#D22828', label='Major roads'),
    Patch(facecolor='#285AC8', label='Local roads'),
    Patch(facecolor='#28AA46', label='Minor roads'),
]
fig.legend(handles=legend, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, 0.0),
           frameon=True, edgecolor='grey')

fig.suptitle(
    'Error Analysis — Representative Failure and Success Cases\n'
    'SegFormer-B2 vs U-Net/ResNet50 | 3-Class Scheme',
    fontsize=13, fontweight='bold', y=0.995)

fig.subplots_adjust(top=0.96, bottom=0.04,
                    left=0.02, right=0.98,
                    hspace=0.08, wspace=0.03)
combined_path = OUT_DIR / 'artifact3_all_cases_combined.png'
plt.savefig(str(combined_path), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*70}")
print(f"ALL OUTPUTS SAVED TO: {OUT_DIR}")
print(f"{'='*70}")
print(summary_df.to_string(index=False))