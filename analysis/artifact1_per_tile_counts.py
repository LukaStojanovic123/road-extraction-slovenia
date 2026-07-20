"""
artifact1_per_tile_counts.py
Generates per-tile raw pixel counts (TP/FP/FN) for all test tiles,
all models, both class schemes.
Output: outputs/article/artifact1_per_tile_counts.csv
This single file enables:
  - Bootstrap confidence intervals
  - Paired model comparison with CIs
  - Per-landscape uncertainty quantification
  - Objective failure case selection for Artifact 3
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# ── Paths ──────────────────────────────────────────────────────────────────
TILE_INDEX_3  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_road_only.csv")
TILE_INDEX_2  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_2class.csv")
MODELS_3CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models")
MODELS_2CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models\2class")
OUT_DIR       = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']

# ── Transform ──────────────────────────────────────────────────────────────
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ── Dataset ────────────────────────────────────────────────────────────────
class RoadDataset(Dataset):
    def __init__(self, tile_df, transform=None):
        self.tiles     = tile_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row   = self.tiles.iloc[idx]
        image = np.load(row['image_path'])
        mask  = np.load(row['mask_path'])
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask']
        return (image, mask.long(),
                row['tile_id'],
                row['municipality'],
                row['landscape_type'])

# ── Models ─────────────────────────────────────────────────────────────────
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
    return model

def compute_tile_counts(preds, targets, num_road_classes):
    """
    Compute TP, FP, FN per road class for a batch.
    Returns list of dicts, one per sample in batch.
    num_road_classes: number of road classes (3 for 3-class, 2 for 2-class)
    """
    preds   = preds.argmax(dim=1)
    results = []
    for i in range(preds.shape[0]):
        p = preds[i].cpu().numpy().flatten()
        t = targets[i].cpu().numpy().flatten()
        row = {}
        for cls in range(1, num_road_classes + 1):
            p_cls = (p == cls)
            t_cls = (t == cls)
            row[f'tp_cls{cls}'] = int((p_cls &  t_cls).sum())
            row[f'fp_cls{cls}'] = int((p_cls & ~t_cls).sum())
            row[f'fn_cls{cls}'] = int((~p_cls & t_cls).sum())
        results.append(row)
    return results

def run_inference(tile_df, model, num_classes, num_road_classes,
                  model_name, scheme, batch_size=64):
    dataset = RoadDataset(tile_df, transform=test_transform)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0,
                         pin_memory=False)

    records = []
    t0      = time.time()

    with torch.no_grad():
        for batch_idx, (images, masks, tile_ids,
                        municipalities, landscapes) in enumerate(loader):
            images  = images.cuda()
            masks   = masks.cuda()
            outputs = model(images)
            counts  = compute_tile_counts(outputs, masks, num_road_classes)

            for i, cnt in enumerate(counts):
                rec = {
                    'tile_id':      tile_ids[i],
                    'municipality': municipalities[i],
                    'landscape':    landscapes[i],
                    'model':        model_name,
                    'scheme':       scheme,
                }
                rec.update(cnt)
                records.append(rec)

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate    = (batch_idx + 1) * batch_size / elapsed
                remain  = (len(loader) - batch_idx - 1) * batch_size / rate / 60
                print(f"    Batch {batch_idx+1}/{len(loader)} | "
                      f"{rate:.0f} tiles/s | ~{remain:.1f} min remaining",
                      flush=True)

    return records


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
print("="*70)
print("ARTIFACT 1 — PER-TILE PIXEL COUNTS")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load test tiles
tile_df_3 = pd.read_csv(TILE_INDEX_3)
tile_df_2 = pd.read_csv(TILE_INDEX_2)
test_df_3 = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)
test_df_2 = tile_df_2[tile_df_2['split'] == 'test'].reset_index(drop=True)

print(f"\nTest tiles (3-class): {len(test_df_3):,}")
print(f"Test tiles (2-class): {len(test_df_2):,}")

all_records = []

# ── 3-class inference ──────────────────────────────────────────────────────
print("\n=== 3-class inference ===")
for model_name in MODEL_NAMES:
    print(f"\n  {model_name}...")
    model = load_model(model_name, MODELS_3CLASS, num_classes=4)
    recs  = run_inference(
        test_df_3, model,
        num_classes=4, num_road_classes=3,
        model_name=model_name, scheme='3class')
    all_records.extend(recs)
    del model
    torch.cuda.empty_cache()
    print(f"  Done — {len(recs):,} tiles")

# ── 2-class inference ──────────────────────────────────────────────────────
print("\n=== 2-class inference ===")
for model_name in MODEL_NAMES:
    print(f"\n  {model_name}...")
    model = load_model(model_name, MODELS_2CLASS, num_classes=3)
    recs  = run_inference(
        test_df_2, model,
        num_classes=3, num_road_classes=2,
        model_name=model_name, scheme='2class')
    all_records.extend(recs)
    del model
    torch.cuda.empty_cache()
    print(f"  Done — {len(recs):,} tiles")

# ── Save raw counts ────────────────────────────────────────────────────────
counts_df = pd.DataFrame(all_records)

# Rename columns to match requested schema
# 3-class: tp_cls1=tp_major, tp_cls2=tp_local, tp_cls3=tp_minor
# 2-class: tp_cls1=tp_primary, tp_cls2=tp_secondary
def rename_cols(df):
    mask_3 = df['scheme'] == '3class'
    mask_2 = df['scheme'] == '2class'
    for stat in ['tp', 'fp', 'fn']:
        # 3-class
        df.loc[mask_3, f'{stat}_major'] = df.loc[mask_3, f'{stat}_cls1']
        df.loc[mask_3, f'{stat}_local'] = df.loc[mask_3, f'{stat}_cls2']
        df.loc[mask_3, f'{stat}_minor'] = df.loc[mask_3, f'{stat}_cls3']
        # 2-class
        df.loc[mask_2, f'{stat}_primary']   = df.loc[mask_2, f'{stat}_cls1']
        df.loc[mask_2, f'{stat}_secondary'] = df.loc[mask_2, f'{stat}_cls2']
    return df

counts_df = rename_cols(counts_df)

# Drop internal cls columns
drop_cols = [c for c in counts_df.columns
             if c.startswith('tp_cls') or
                c.startswith('fp_cls') or
                c.startswith('fn_cls')]
counts_df = counts_df.drop(columns=drop_cols)

out_path = OUT_DIR / 'artifact1_per_tile_counts.csv'
counts_df.to_csv(out_path, index=False)

print(f"\n{'='*70}")
print(f"Saved: {out_path}")
print(f"Total rows: {len(counts_df):,}")
print(f"Columns: {counts_df.columns.tolist()}")

# ── Quick verification — recompute overall F1 from counts ──────────────────
print("\n=== Verification: recomputed Road Macro F1 from raw counts ===")
print("(should match test_v2.py and test_2class.py results)\n")

for scheme in ['3class', '2class']:
    sdf = counts_df[counts_df['scheme'] == scheme]
    cls_cols = (['major','local','minor'] if scheme == '3class'
                else ['primary','secondary'])
    print(f"  {scheme}:")
    for model_name in MODEL_NAMES:
        mdf = sdf[sdf['model'] == model_name]
        road_f1 = []
        for cls in cls_cols:
            tp = mdf[f'tp_{cls}'].sum()
            fp = mdf[f'fp_{cls}'].sum()
            fn = mdf[f'fn_{cls}'].sum()
            prec = tp / (tp + fp + 1e-6)
            rec  = tp / (tp + fn + 1e-6)
            f1   = 2 * prec * rec / (prec + rec + 1e-6)
            road_f1.append(f1)
        print(f"    {model_name:20s} Road Macro F1 = {np.mean(road_f1):.4f}")
    print()