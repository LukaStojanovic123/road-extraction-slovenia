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
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# Paths
TILE_INDEX  = ROOT_DIR / "data/processed/metadata/tile_index_road_only.csv"
MODELS_DIR  = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "outputs/metrics"
FIGURES_DIR = ROOT_DIR / "outputs/figures"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transform
test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Dataset
class RoadDataset(Dataset):
    def __init__(self, tile_df, transform=None):
        self.tiles = tile_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row = self.tiles.iloc[idx]
        image = np.load(row['image_path'])
        mask  = np.load(row['mask_path'])
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask']
        return image, mask.long(), row['municipality'], row['landscape_type']

# Models
def get_model(model_name):
    if model_name == 'unet_resnet50':
        return smp.Unet(
            encoder_name='resnet50', encoder_weights=None,
            in_channels=3, classes=4)
    elif model_name == 'dlinknet':
        return smp.Unet(
            encoder_name='resnet34', encoder_weights=None,
            in_channels=3, classes=4, decoder_use_batchnorm=True)
    elif model_name == 'segformer':
        return smp.Segformer(
            encoder_name='mit_b2', encoder_weights=None,
            in_channels=3, classes=4)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name='resnet50', encoder_weights=None,
            in_channels=3, classes=4)

def load_model(model_name):
    model = get_model(model_name).cuda()
    checkpoint = torch.load(
        MODELS_DIR / model_name / f"{model_name}_best.pth",
        map_location='cuda',
        weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['epoch']

# Metrics
def calculate_metrics(preds, targets, num_classes=4):
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)
    metrics = {}
    road_f1  = []
    road_iou = []
    for cls in range(num_classes):
        pred_cls   = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()
        iou = (intersection / (union + 1e-6)).item()
        tp = intersection.item()
        fp = (pred_cls & ~target_cls).sum().float().item()
        fn = (~pred_cls & target_cls).sum().float().item()
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        metrics[f'iou_class_{cls}']       = iou
        metrics[f'f1_class_{cls}']        = f1
        metrics[f'precision_class_{cls}'] = precision
        metrics[f'recall_class_{cls}']    = recall
        if cls > 0:
            road_f1.append(f1)
            road_iou.append(iou)
    metrics['miou']          = np.mean([metrics[f'iou_class_{c}'] for c in range(num_classes)])
    metrics['macro_f1']      = np.mean([metrics[f'f1_class_{c}'] for c in range(num_classes)])
    metrics['road_macro_f1'] = np.mean(road_f1)
    metrics['road_miou']     = np.mean(road_iou)
    return metrics

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [240, 240, 240]
    rgb[mask == 1] = [255, 0,   0]
    rgb[mask == 2] = [0,   0,   255]
    rgb[mask == 3] = [0,   200, 0]
    return rgb

def predict_single(model, img):
    augmented = test_transform(image=img,
                               mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = augmented['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1).squeeze().cpu().numpy()

# Full evaluation
def evaluate_model(model_name, model, test_df):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    test_dataset = RoadDataset(test_df, transform=test_transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=64,
        shuffle=False, num_workers=4, pin_memory=False)

    print(f"Test tiles: {len(test_dataset):,}")

    all_metrics       = []
    muni_metrics      = {}
    landscape_metrics = {}

    with torch.no_grad():
        for batch_idx, (images, masks, municipalities, landscapes) in enumerate(test_loader):
            images = images.cuda()
            masks  = masks.cuda()
            outputs = model(images)

            batch_metrics = calculate_metrics(outputs, masks)
            all_metrics.append(batch_metrics)

            for i in range(len(images)):
                muni = municipalities[i]
                land = landscapes[i]

                single_pred = outputs[i:i+1]
                single_mask = masks[i:i+1]
                sm = calculate_metrics(single_pred, single_mask)

                if muni not in muni_metrics:
                    muni_metrics[muni] = []
                muni_metrics[muni].append(sm)

                if land not in landscape_metrics:
                    landscape_metrics[land] = []
                landscape_metrics[land].append(sm)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}", flush=True)

    # Average overall
    overall = {k: np.mean([m[k] for m in all_metrics])
               for k in all_metrics[0].keys()}

    print(f"\n--- Overall Test Results ---")
    print(f"Road Macro F1:   {overall['road_macro_f1']:.4f}")
    print(f"Road mIoU:       {overall['road_miou']:.4f}")
    print(f"mIoU:            {overall['miou']:.4f}")
    print(f"F1 Major:        {overall['f1_class_1']:.4f}")
    print(f"F1 Local:        {overall['f1_class_2']:.4f}")
    print(f"F1 Minor:        {overall['f1_class_3']:.4f}")
    print(f"Precision Major: {overall['precision_class_1']:.4f}")
    print(f"Recall Major:    {overall['recall_class_1']:.4f}")

    # Per municipality
    print(f"\n--- Per Municipality ---")
    muni_rows = []
    for muni, mlist in sorted(muni_metrics.items()):
        avg = {k: np.mean([m[k] for m in mlist]) for k in mlist[0].keys()}
        print(f"  {muni:25s} F1={avg['road_macro_f1']:.4f} "
              f"Major={avg['f1_class_1']:.4f} "
              f"Local={avg['f1_class_2']:.4f} "
              f"Minor={avg['f1_class_3']:.4f}")
        muni_rows.append({'municipality': muni, 'model': model_name, **avg})

    # Per landscape
    print(f"\n--- Per Landscape Type ---")
    land_rows = []
    for land, llist in sorted(landscape_metrics.items()):
        avg = {k: np.mean([m[k] for m in llist]) for k in llist[0].keys()}
        print(f"  {land:20s} F1={avg['road_macro_f1']:.4f} "
              f"Major={avg['f1_class_1']:.4f} "
              f"Local={avg['f1_class_2']:.4f} "
              f"Minor={avg['f1_class_3']:.4f}")
        land_rows.append({'landscape_type': land, 'model': model_name, **avg})

    # Save CSVs
    pd.DataFrame([{'model': model_name, **overall}]).to_csv(
        METRICS_DIR / f"{model_name}_test_overall.csv", index=False)
    pd.DataFrame(muni_rows).to_csv(
        METRICS_DIR / f"{model_name}_test_per_municipality.csv", index=False)
    pd.DataFrame(land_rows).to_csv(
        METRICS_DIR / f"{model_name}_test_per_landscape.csv", index=False)

    return overall, muni_rows, land_rows


# Visual preview
def generate_visual(models_dict, test_df):
    print("\nGenerating visual predictions...")
    municipalities = sorted(test_df['municipality'].unique())
    model_names    = list(models_dict.keys())

    fig, axes = plt.subplots(
        len(municipalities), 6,
        figsize=(24, 4 * len(municipalities))
    )

    col_titles = ['Orthophoto', 'Ground Truth',
                  'U-Net ResNet50', 'D-LinkNet',
                  'SegFormer', 'DeepLabV3+']

    for i, muni in enumerate(municipalities):
        muni_tiles = test_df[test_df['municipality'] == muni]
        best_tile  = muni_tiles.loc[muni_tiles['road_pixel_ratio'].idxmax()]

        img  = np.load(best_tile['image_path'])
        mask = np.load(best_tile['mask_path'])

        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(muni, fontsize=9, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_to_rgb(mask))
        axes[i, 1].axis('off')

        for j, model_name in enumerate(model_names):
            pred = predict_single(models_dict[model_name], img)
            axes[i, j+2].imshow(mask_to_rgb(pred))
            axes[i, j+2].axis('off')

        print(f"  {muni} done", flush=True)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight='bold')

    legend_elements = [
        Patch(facecolor='#F0F0F0', label='Background'),
        Patch(facecolor='#FF0000', label='Major roads'),
        Patch(facecolor='#0000FF', label='Local roads'),
        Patch(facecolor='#00C800', label='Minor roads')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Test Municipality Predictions — All Models',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / 'test_predictions_all_models.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load test tiles
    tile_df  = pd.read_csv(TILE_INDEX)
    test_df  = tile_df[tile_df['split'] == 'test'].reset_index(drop=True)
    print(f"Test tiles:          {len(test_df):,}")
    print(f"Test municipalities: {test_df['municipality'].unique().tolist()}")

    model_names = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']

    # Load all models
    print("\nLoading models...")
    models_dict = {}
    for name in model_names:
        model, epoch = load_model(name)
        models_dict[name] = model
        print(f"  {name} loaded (best epoch {epoch})")

    # Evaluate all models
    all_overall  = {}
    all_muni     = []
    all_landscape = []

    for model_name in model_names:
        overall, muni_rows, land_rows = evaluate_model(
            model_name, models_dict[model_name], test_df)
        all_overall[model_name] = overall
        all_muni.extend(muni_rows)
        all_landscape.extend(land_rows)
        torch.cuda.empty_cache()

    # Generate visual
    generate_visual(models_dict, test_df)

    # Final comparison table
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Road F1':>10} {'Major F1':>10} "
          f"{'Local F1':>10} {'Minor F1':>10} {'mIoU':>10}")
    print("-"*60)
    for name, res in all_overall.items():
        print(f"{name:<20} {res['road_macro_f1']:>10.4f} "
              f"{res['f1_class_1']:>10.4f} "
              f"{res['f1_class_2']:>10.4f} "
              f"{res['f1_class_3']:>10.4f} "
              f"{res['miou']:>10.4f}")

    # Save combined CSVs
    pd.DataFrame(list(all_overall.items()),
                 columns=['model', 'metrics']).to_csv(
        METRICS_DIR / 'test_results_all_models.csv', index=False)
    pd.concat([pd.DataFrame(all_muni)]).to_csv(
        METRICS_DIR / 'test_results_per_municipality.csv', index=False)
    pd.concat([pd.DataFrame(all_landscape)]).to_csv(
        METRICS_DIR / 'test_results_per_landscape.csv', index=False)

    print(f"\nAll results saved to: {METRICS_DIR}")
    print(f"Figures saved to:     {FIGURES_DIR}")