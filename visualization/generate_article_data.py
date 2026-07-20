"""
generate_article_data.py
Generates all tables, figures, and statistics needed for the ISPRS paper.
Run AFTER test_v2.py and test_2class.py have completed.
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
METRICS_3CLASS  = ROOT_DIR / "outputs/metrics"
METRICS_2CLASS  = ROOT_DIR / "outputs/2class/metrics"
TILE_INDEX_3    = ROOT_DIR / "data/processed/metadata/tile_index_road_only.csv"
TILE_INDEX_2    = ROOT_DIR / "data/processed/metadata/tile_index_2class.csv"
OUT_DIR         = ROOT_DIR / "outputs/article"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES  = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']
MODEL_LABELS = {
    'unet_resnet50':  'U-Net/ResNet50',
    'dlinknet':       'D-LinkNet',
    'segformer':      'SegFormer-B2',
    'deeplabv3plus':  'DeepLabV3+'
}

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       10,
    'axes.titlesize':  11,
    'axes.labelsize':  10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight'
})

COLORS = {
    'unet_resnet50':  '#2196F3',
    'dlinknet':       '#FF9800',
    'segformer':      '#4CAF50',
    'deeplabv3plus':  '#E91E63'
}

print("="*70)
print("ARTICLE DATA GENERATOR")
print("="*70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET STATISTICS TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/10] Generating dataset statistics table...")

tile_df = pd.read_csv(TILE_INDEX_3)

dataset_stats = []
for split in ['train', 'val', 'test']:
    sdf = tile_df[tile_df['split'] == split]
    dataset_stats.append({
        'Split':            split.capitalize(),
        'Municipalities':   sdf['municipality'].nunique(),
        'Tiles':            f"{len(sdf):,}",
        'Background (%)':   f"{sdf['background_ratio'].mean()*100:.1f}",
        'Major roads (%)':  f"{sdf['class_1_ratio'].mean()*100:.2f}",
        'Local roads (%)':  f"{sdf['class_2_ratio'].mean()*100:.2f}",
        'Minor roads (%)':  f"{sdf['class_3_ratio'].mean()*100:.2f}",
        'Road total (%)':   f"{sdf['road_pixel_ratio'].mean()*100:.2f}"
    })

dataset_df = pd.DataFrame(dataset_stats)
dataset_df.to_csv(OUT_DIR / 'table_dataset_statistics.csv', index=False)
print("  Saved: table_dataset_statistics.csv")
print(dataset_df.to_string(index=False))

# Per-municipality overview
muni_overview = tile_df.groupby(
    ['split', 'municipality', 'landscape_type']
).agg(
    tiles=('tile_id', 'count'),
    road_ratio=('road_pixel_ratio', 'mean')
).reset_index()
muni_overview.to_csv(OUT_DIR / 'table_municipality_split_overview.csv', index=False)
print("  Saved: table_municipality_split_overview.csv")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — 3-CLASS OVERALL TEST RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/10] Generating 3-class overall results table...")

rows_3class = []
for model in MODEL_NAMES:
    csv = METRICS_3CLASS / f"{model}_test_overall_v2.csv"
    if not csv.exists():
        print(f"  WARNING: {csv} not found — skipping")
        continue
    df = pd.read_csv(csv)
    rows_3class.append({
        'Model':             MODEL_LABELS[model],
        'Precision Major':   f"{df['precision_class_1'].values[0]:.4f}",
        'Recall Major':      f"{df['recall_class_1'].values[0]:.4f}",
        'F1 Major':          f"{df['f1_class_1'].values[0]:.4f}",
        'IoU Major':         f"{df['iou_class_1'].values[0]:.4f}",
        'Precision Local':   f"{df['precision_class_2'].values[0]:.4f}",
        'Recall Local':      f"{df['recall_class_2'].values[0]:.4f}",
        'F1 Local':          f"{df['f1_class_2'].values[0]:.4f}",
        'IoU Local':         f"{df['iou_class_2'].values[0]:.4f}",
        'Precision Minor':   f"{df['precision_class_3'].values[0]:.4f}",
        'Recall Minor':      f"{df['recall_class_3'].values[0]:.4f}",
        'F1 Minor':          f"{df['f1_class_3'].values[0]:.4f}",
        'IoU Minor':         f"{df['iou_class_3'].values[0]:.4f}",
        'Road Macro F1':     f"{df['road_macro_f1'].values[0]:.4f}",
        'Road mIoU':         f"{df['road_miou'].values[0]:.4f}",
        'mIoU':              f"{df['miou'].values[0]:.4f}",
    })

results_3class_df = pd.DataFrame(rows_3class)
results_3class_df.to_csv(OUT_DIR / 'table_3class_overall_results.csv', index=False)
print("  Saved: table_3class_overall_results.csv")
print(results_3class_df[['Model', 'F1 Major', 'F1 Local', 'F1 Minor',
                          'Road Macro F1', 'mIoU']].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — 2-CLASS OVERALL TEST RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/10] Generating 2-class overall results table...")

rows_2class = []
for model in MODEL_NAMES:
    csv = METRICS_2CLASS / f"{model}_test_overall.csv"
    if not csv.exists():
        print(f"  WARNING: {csv} not found — skipping")
        continue
    df = pd.read_csv(csv)
    rows_2class.append({
        'Model':               MODEL_LABELS[model],
        'Precision Primary':   f"{df['precision_class_1'].values[0]:.4f}",
        'Recall Primary':      f"{df['recall_class_1'].values[0]:.4f}",
        'F1 Primary':          f"{df['f1_class_1'].values[0]:.4f}",
        'IoU Primary':         f"{df['iou_class_1'].values[0]:.4f}",
        'Precision Secondary': f"{df['precision_class_2'].values[0]:.4f}",
        'Recall Secondary':    f"{df['recall_class_2'].values[0]:.4f}",
        'F1 Secondary':        f"{df['f1_class_2'].values[0]:.4f}",
        'IoU Secondary':       f"{df['iou_class_2'].values[0]:.4f}",
        'Road Macro F1':       f"{df['road_macro_f1'].values[0]:.4f}",
        'Road mIoU':           f"{df['road_miou'].values[0]:.4f}",
        'mIoU':                f"{df['miou'].values[0]:.4f}",
    })

results_2class_df = pd.DataFrame(rows_2class)
results_2class_df.to_csv(OUT_DIR / 'table_2class_overall_results.csv', index=False)
print("  Saved: table_2class_overall_results.csv")
print(results_2class_df[['Model', 'F1 Primary', 'F1 Secondary',
                          'Road Macro F1', 'mIoU']].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — 3-CLASS vs 2-CLASS COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/10] Generating 3-class vs 2-class comparison table...")

comparison_rows = []
for model in MODEL_NAMES:
    csv3 = METRICS_3CLASS / f"{model}_test_overall_v2.csv"
    csv2 = METRICS_2CLASS / f"{model}_test_overall.csv"
    if not csv3.exists() or not csv2.exists():
        continue
    df3  = pd.read_csv(csv3)
    df2  = pd.read_csv(csv2)
    f1_3  = df3['road_macro_f1'].values[0]
    f1_2  = df2['road_macro_f1'].values[0]
    iou_3 = df3['road_miou'].values[0]
    iou_2 = df2['road_miou'].values[0]
    comparison_rows.append({
        'Model':             MODEL_LABELS[model],
        '3-class Road F1':   f"{f1_3:.4f}",
        '2-class Road F1':   f"{f1_2:.4f}",
        'ΔF1 (3→2)':        f"{f1_2 - f1_3:+.4f}",
        '3-class Road mIoU': f"{iou_3:.4f}",
        '2-class Road mIoU': f"{iou_2:.4f}",
        'ΔmIoU (3→2)':      f"{iou_2 - iou_3:+.4f}",
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(OUT_DIR / 'table_3class_vs_2class_comparison.csv', index=False)
print("  Saved: table_3class_vs_2class_comparison.csv")
print(comparison_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER LANDSCAPE TYPE TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/10] Generating per-landscape analysis...")

landscape_rows = []
for model in MODEL_NAMES:
    csv = METRICS_3CLASS / f"{model}_test_per_landscape_v2.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        landscape_rows.append({
            'Model':       MODEL_LABELS[model],
            'Landscape':   row['landscape_type'].capitalize(),
            'Road F1':     row['road_macro_f1'],
            'F1 Major':    row['f1_class_1'],
            'F1 Local':    row['f1_class_2'],
            'F1 Minor':    row['f1_class_3'],
            'Road mIoU':   row['road_miou']
        })

landscape_df = pd.DataFrame(landscape_rows)
landscape_df.to_csv(OUT_DIR / 'table_per_landscape_results.csv', index=False)
print("  Saved: table_per_landscape_results.csv")

pivot = landscape_df.pivot_table(
    index='Landscape', columns='Model',
    values='Road F1', aggfunc='first')
print(pivot.round(4).to_string())

# Per municipality table
muni_rows = []
for model in MODEL_NAMES:
    csv = METRICS_3CLASS / f"{model}_test_per_municipality_v2.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        muni_rows.append({
            'Model':      MODEL_LABELS[model],
            'Municipality': row['municipality'],
            'Road F1':    row['road_macro_f1'],
            'F1 Major':   row['f1_class_1'],
            'F1 Local':   row['f1_class_2'],
            'F1 Minor':   row['f1_class_3'],
            'Road mIoU':  row['road_miou']
        })

muni_df = pd.DataFrame(muni_rows)
muni_df.to_csv(OUT_DIR / 'table_per_municipality_results.csv', index=False)
print("  Saved: table_per_municipality_results.csv")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING CURVES FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/10] Generating training curves figure...")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for col, model in enumerate(MODEL_NAMES):
    csv = METRICS_3CLASS / f"{model}_history.csv"
    if not csv.exists():
        print(f"  WARNING: {csv} not found — skipping")
        continue
    df     = pd.read_csv(csv)
    epochs = df['epoch']
    color  = COLORS[model]

    # Handle both old and new column naming
    metric_col = 'road_macro_f1' if 'road_macro_f1' in df.columns \
                 else 'road_miou'
    best_epoch = df.loc[df[metric_col].idxmax(), 'epoch']

    # Row 0 — Loss curves
    axes[0, col].plot(epochs, df['train_loss'],
                      color=color, linestyle='-',
                      linewidth=1.5, label='Train')
    axes[0, col].plot(epochs, df['val_loss'],
                      color=color, linestyle='--',
                      linewidth=1.5, label='Val')
    axes[0, col].axvline(x=best_epoch, color='red',
                          linestyle=':', alpha=0.7, linewidth=1,
                          label=f'Best (ep.{int(best_epoch)})')
    axes[0, col].set_title(MODEL_LABELS[model], fontweight='bold')
    axes[0, col].set_xlabel('Epoch')
    if col == 0:
        axes[0, col].set_ylabel('Loss')
    axes[0, col].legend(fontsize=7)
    axes[0, col].grid(True, alpha=0.3)

    # Row 1 — F1 curves per class
    if 'f1_class_1' in df.columns:
        axes[1, col].plot(epochs, df['f1_class_1'],
                          color='#E53935', linewidth=1.5, label='Major')
        axes[1, col].plot(epochs, df['f1_class_2'],
                          color='#1E88E5', linewidth=1.5, label='Local')
        axes[1, col].plot(epochs, df['f1_class_3'],
                          color='#43A047', linewidth=1.5, label='Minor')
    axes[1, col].plot(epochs, df[metric_col],
                      color='black', linewidth=2,
                      linestyle='--', label='Road Macro F1')
    axes[1, col].axvline(x=best_epoch, color='red',
                          linestyle=':', alpha=0.7, linewidth=1)
    axes[1, col].set_xlabel('Epoch')
    if col == 0:
        axes[1, col].set_ylabel('F1 Score')
    axes[1, col].legend(fontsize=7)
    axes[1, col].grid(True, alpha=0.3)
    axes[1, col].set_ylim(0, 1)

fig.suptitle('Training and Validation Curves — 3-Class Models',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_training_curves_3class.png')
plt.close()
print("  Saved: fig_training_curves_3class.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — PER LANDSCAPE BAR CHART
# ══════════════════════════════════════════════════════════════════════════
print("\n[7/10] Generating per-landscape bar chart...")

landscapes = sorted(landscape_df['Landscape'].unique())
x          = np.arange(len(landscapes))
width      = 0.2

fig, ax = plt.subplots(figsize=(14, 6))

for i, model in enumerate(MODEL_NAMES):
    mdf    = landscape_df[landscape_df['Model'] == MODEL_LABELS[model]]
    values = []
    for l in landscapes:
        row = mdf[mdf['Landscape'] == l]
        values.append(row['Road F1'].values[0] if len(row) > 0 else 0)

    bars = ax.bar(x + i*width, values, width,
                  label=MODEL_LABELS[model],
                  color=COLORS[model],
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=90)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(landscapes, fontsize=10)
ax.set_ylabel('Road Macro F1 Score')
ax.set_title('Model Performance by Landscape Type — Test Set (3-class)',
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 0.85)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_per_landscape_f1_3class.png')
plt.close()
print("  Saved: fig_per_landscape_f1_3class.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — 3-CLASS VS 2-CLASS COMPARISON FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n[8/10] Generating 3-class vs 2-class comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(MODEL_NAMES))

for ax_idx, (metric_key, ax, title) in enumerate(zip(
        ['road_macro_f1', 'road_miou'],
        axes,
        ['Road Macro F1', 'Road mIoU'])):

    vals_3, vals_2 = [], []
    for model in MODEL_NAMES:
        csv3 = METRICS_3CLASS / f"{model}_test_overall_v2.csv"
        csv2 = METRICS_2CLASS / f"{model}_test_overall.csv"
        v3 = pd.read_csv(csv3)[metric_key].values[0] if csv3.exists() else 0
        v2 = pd.read_csv(csv2)[metric_key].values[0] if csv2.exists() else 0
        vals_3.append(v3)
        vals_2.append(v2)

    bars1 = ax.bar(x - 0.2, vals_3, 0.35,
                   label='3-class', color='#1565C0',
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + 0.2, vals_2, 0.35,
                   label='2-class', color='#E53935',
                   edgecolor='white', linewidth=0.5)

    for bar, val in zip(list(bars1) + list(bars2), vals_3 + vals_2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES],
                       rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(title)
    ax.set_title(f"{title} — 3-class vs 2-class", fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.85)

plt.suptitle('3-Class vs 2-Class Model Comparison — Test Set',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_3class_vs_2class_comparison.png')
plt.close()
print("  Saved: fig_3class_vs_2class_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRECISION-RECALL SCATTER FIGURE
# ══════════════════════════════════════════════════════════════════════════
print("\n[9/10] Generating precision-recall figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
class_info = [
    (1, 'Major roads', '#E53935'),
    (2, 'Local roads', '#1E88E5'),
    (3, 'Minor roads', '#43A047')
]

for ax, (cls, cls_name, cls_color) in zip(axes, class_info):
    for model in MODEL_NAMES:
        csv = METRICS_3CLASS / f"{model}_test_overall_v2.csv"
        if not csv.exists():
            continue
        df   = pd.read_csv(csv)
        prec = df[f'precision_class_{cls}'].values[0]
        rec  = df[f'recall_class_{cls}'].values[0]
        f1   = df[f'f1_class_{cls}'].values[0]
        ax.scatter(rec, prec, s=200,
                   color=COLORS[model], zorder=5,
                   label=f"{MODEL_LABELS[model]} (F1={f1:.3f})")
        ax.annotate(MODEL_LABELS[model].split('/')[0],
                    (rec, prec),
                    textcoords='offset points',
                    xytext=(6, 4), fontsize=8)

    # F1 iso-curves
    for f1_val in [0.4, 0.5, 0.6, 0.7, 0.8]:
        r_range = np.linspace(0.01, 1.0, 300)
        denom   = 2 * r_range - f1_val
        with np.errstate(divide='ignore', invalid='ignore'):
            p_range = np.where(denom > 0,
                                f1_val * r_range / denom,
                                np.nan)
        mask = (p_range > 0) & (p_range <= 1.0)
        ax.plot(r_range[mask], p_range[mask],
                'k--', alpha=0.2, linewidth=0.8)
        if mask.any():
            mid = len(r_range[mask]) // 2
            ax.annotate(f'F1={f1_val}',
                        (r_range[mask][mid], p_range[mask][mid]),
                        fontsize=7, color='grey')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(cls_name, fontweight='bold', color=cls_color)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='lower left')

plt.suptitle('Precision-Recall Analysis per Road Class — Test Set (3-class)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_precision_recall_per_class.png')
plt.close()
print("  Saved: fig_precision_recall_per_class.png")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — MUNICIPALITY HEATMAP + CLASS DISTRIBUTION + SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n[10/10] Generating heatmap, class distribution, and summary...")

# Municipality heatmap
muni_data = {}
for model in MODEL_NAMES:
    csv = METRICS_3CLASS / f"{model}_test_per_municipality_v2.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    for _, row in df.iterrows():
        muni = row['municipality']
        if muni not in muni_data:
            muni_data[muni] = {}
        muni_data[muni][MODEL_LABELS[model]] = row['road_macro_f1']

if muni_data:
    heatmap_df = pd.DataFrame(muni_data).T
    heatmap_df.index.name = 'Municipality'

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(heatmap_df.values, cmap='RdYlGn',
                   aspect='auto', vmin=0.3, vmax=0.8)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns,
                       rotation=15, ha='right', fontsize=10)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, fontsize=10)
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.values[i, j]
            ax.text(j, i, f'{val:.3f}',
                    ha='center', va='center',
                    color='black' if 0.4 < val < 0.75 else 'white',
                    fontsize=9, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Road Macro F1')
    ax.set_title(
        'Road Macro F1 per Test Municipality — All Models (3-class)',
        fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_municipality_heatmap.png')
    plt.close()
    print("  Saved: fig_municipality_heatmap.png")

# Class distribution pie charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
splits     = ['train', 'val', 'test']
colors_pie = ['#B0BEC5', '#E53935', '#1E88E5', '#43A047']
labels_pie = ['Background', 'Major', 'Local', 'Minor']

for ax, split in zip(axes, splits):
    sdf    = tile_df[tile_df['split'] == split]
    values = [
        sdf['background_ratio'].mean(),
        sdf['class_1_ratio'].mean(),
        sdf['class_2_ratio'].mean(),
        sdf['class_3_ratio'].mean()
    ]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels_pie, colors=colors_pie,
        autopct='%1.2f%%', startangle=90,
        textprops={'fontsize': 9})
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(
        f"{split.capitalize()} split\n"
        f"({sdf['municipality'].nunique()} municipalities, "
        f"{len(sdf):,} tiles)",
        fontweight='bold')

plt.suptitle('Class Distribution per Split — 3-Class Dataset',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_class_distribution.png')
plt.close()
print("  Saved: fig_class_distribution.png")

# ── Paper writing summary ──────────────────────────────────────────────────
summary_lines = []
summary_lines.append("="*70)
summary_lines.append("PAPER WRITING SUMMARY — KEY NUMBERS FOR ISPRS SUBMISSION")
summary_lines.append("="*70)

summary_lines.append("\n--- DATASET ---")
summary_lines.append(f"Total tiles:              {len(tile_df):,}")
summary_lines.append(f"Train tiles:              "
                     f"{len(tile_df[tile_df['split']=='train']):,}")
summary_lines.append(f"Val tiles:                "
                     f"{len(tile_df[tile_df['split']=='val']):,}")
summary_lines.append(f"Test tiles:               "
                     f"{len(tile_df[tile_df['split']=='test']):,}")
summary_lines.append(f"Tile size:                512x512 px at 0.25m = 128x128m")
summary_lines.append(f"Train municipalities:     12 (7 landscape types)")
summary_lines.append(f"Val municipalities:       3")
summary_lines.append(f"Test municipalities:      5")
summary_lines.append(f"Background pixels avg:    "
                     f"{tile_df['background_ratio'].mean()*100:.2f}%")
summary_lines.append(f"Major road pixels avg:    "
                     f"{tile_df['class_1_ratio'].mean()*100:.3f}%")
summary_lines.append(f"Local road pixels avg:    "
                     f"{tile_df['class_2_ratio'].mean()*100:.3f}%")
summary_lines.append(f"Minor road pixels avg:    "
                     f"{tile_df['class_3_ratio'].mean()*100:.3f}%")
summary_lines.append(f"Road pixels avg:          "
                     f"{tile_df['road_pixel_ratio'].mean()*100:.3f}%")

summary_lines.append("\n--- 3-CLASS TEST RESULTS ---")
best_models_3 = {}
for model in MODEL_NAMES:
    csv = METRICS_3CLASS / f"{model}_test_overall_v2.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    best_models_3[model] = df['road_macro_f1'].values[0]
    summary_lines.append(f"\n{MODEL_LABELS[model]}:")
    summary_lines.append(
        f"  Road Macro F1:    {df['road_macro_f1'].values[0]:.4f}")
    summary_lines.append(
        f"  Road mIoU:        {df['road_miou'].values[0]:.4f}")
    summary_lines.append(
        f"  mIoU:             {df['miou'].values[0]:.4f}")
    summary_lines.append(
        f"  F1 Major:         {df['f1_class_1'].values[0]:.4f}  "
        f"(P={df['precision_class_1'].values[0]:.4f} "
        f"R={df['recall_class_1'].values[0]:.4f})")
    summary_lines.append(
        f"  F1 Local:         {df['f1_class_2'].values[0]:.4f}  "
        f"(P={df['precision_class_2'].values[0]:.4f} "
        f"R={df['recall_class_2'].values[0]:.4f})")
    summary_lines.append(
        f"  F1 Minor:         {df['f1_class_3'].values[0]:.4f}  "
        f"(P={df['precision_class_3'].values[0]:.4f} "
        f"R={df['recall_class_3'].values[0]:.4f})")

summary_lines.append("\n--- 2-CLASS TEST RESULTS ---")
for model in MODEL_NAMES:
    csv = METRICS_2CLASS / f"{model}_test_overall.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    summary_lines.append(f"\n{MODEL_LABELS[model]}:")
    summary_lines.append(
        f"  Road Macro F1:    {df['road_macro_f1'].values[0]:.4f}")
    summary_lines.append(
        f"  F1 Primary:       {df['f1_class_1'].values[0]:.4f}  "
        f"(P={df['precision_class_1'].values[0]:.4f} "
        f"R={df['recall_class_1'].values[0]:.4f})")
    summary_lines.append(
        f"  F1 Secondary:     {df['f1_class_2'].values[0]:.4f}  "
        f"(P={df['precision_class_2'].values[0]:.4f} "
        f"R={df['recall_class_2'].values[0]:.4f})")

if best_models_3:
    best = max(best_models_3, key=best_models_3.get)
    summary_lines.append(
        f"\n--- BEST MODEL (3-class) ---\n"
        f"  {MODEL_LABELS[best]} "
        f"(Road Macro F1 = {best_models_3[best]:.4f})")

summary_lines.append("\n--- KEY FINDINGS FOR PAPER ---")
summary_lines.append(
    "1. SegFormer-B2 achieves highest overall Road Macro F1 in 3-class")
summary_lines.append(
    "2. Major roads consistently best detected across all models (wider, more distinct)")
summary_lines.append(
    "3. Minor roads hardest to detect (narrow, under vegetation, label noise)")
summary_lines.append(
    "4. Urban landscapes most challenging for all models")
summary_lines.append(
    "5. 2-class grouping (Primary+Secondary) shows different pattern vs 3-class")
summary_lines.append(
    "6. Spatially separated municipality-level evaluation ensures no data leakage")

summary_txt = '\n'.join(summary_lines)
with open(OUT_DIR / 'paper_writing_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_txt)
print(summary_txt)

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"ALL OUTPUTS SAVED TO: {OUT_DIR}")
print(f"{'='*70}")
print(f"\nFiles generated:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")