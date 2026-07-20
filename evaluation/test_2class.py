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
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Paths
TILE_INDEX  = ROOT_DIR / "data/processed/metadata/tile_index_2class.csv"
MODELS_DIR  = ROOT_DIR / "models/2class"
METRICS_DIR = ROOT_DIR / "outputs/2class/metrics"
FIGURES_DIR = ROOT_DIR / "outputs/2class/figures"
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
        self.tiles     = tile_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row   = self.tiles.iloc[idx]
        image = np.load(row['image_path'])
        mask  = np.load(row['mask_path'])
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image     = augmented['image']
            mask      = augmented['mask']
        return image, mask.long(), row['municipality'], row['landscape_type']

# Models — 3 output classes (bg + primary + secondary)
def get_model(model_name):
    if model_name == 'unet_resnet50':
        return smp.Unet(
            encoder_name='resnet50', encoder_weights=None,
            in_channels=3, classes=3)
    elif model_name == 'dlinknet':
        return smp.Unet(
            encoder_name='resnet34', encoder_weights=None,
            in_channels=3, classes=3, decoder_use_batchnorm=True)
    elif model_name == 'segformer':
        return smp.Segformer(
            encoder_name='mit_b2', encoder_weights=None,
            in_channels=3, classes=3)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name='resnet50', encoder_weights=None,
            in_channels=3, classes=3)

def load_model(model_name):
    model      = get_model(model_name).cuda()
    checkpoint = torch.load(
        MODELS_DIR / model_name / f"{model_name}_best.pth",
        map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['epoch']

# Micro-averaging accumulator
class MetricAccumulator:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.tp = np.zeros(num_classes)
        self.fp = np.zeros(num_classes)
        self.fn = np.zeros(num_classes)

    def update(self, preds, targets):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        for cls in range(self.num_classes):
            pred_cls   = (preds == cls)
            target_cls = (targets == cls)
            self.tp[cls] += ( pred_cls &  target_cls).sum()
            self.fp[cls] += ( pred_cls & ~target_cls).sum()
            self.fn[cls] += (~pred_cls &  target_cls).sum()

    def compute(self):
        metrics  = {}
        road_f1  = []
        road_iou = []
        for cls in range(self.num_classes):
            precision = self.tp[cls] / (self.tp[cls] + self.fp[cls] + 1e-6)
            recall    = self.tp[cls] / (self.tp[cls] + self.fn[cls] + 1e-6)
            f1  = 2 * precision * recall / (precision + recall + 1e-6)
            iou = self.tp[cls] / (self.tp[cls] + self.fp[cls] + self.fn[cls] + 1e-6)
            metrics[f'f1_class_{cls}']        = float(f1)
            metrics[f'iou_class_{cls}']       = float(iou)
            metrics[f'precision_class_{cls}'] = float(precision)
            metrics[f'recall_class_{cls}']    = float(recall)
            if cls > 0:
                road_f1.append(f1)
                road_iou.append(iou)
        metrics['road_macro_f1'] = float(np.mean(road_f1))
        metrics['road_miou']     = float(np.mean(road_iou))
        metrics['miou']          = float(np.mean(
            [metrics[f'iou_class_{c}'] for c in range(self.num_classes)]))
        metrics['macro_f1']      = float(np.mean(
            [metrics[f'f1_class_{c}'] for c in range(self.num_classes)]))
        return metrics

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask == 0] = [240, 240, 240]  # background - grey
    rgb[mask == 1] = [255, 0,   0]    # primary - red
    rgb[mask == 2] = [0,   200, 0]    # secondary - green
    return rgb

def predict_single(model, img):
    aug    = test_transform(image=img,
                            mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = aug['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(tensor)
    return output.argmax(dim=1).squeeze().cpu().numpy()

def evaluate_model(model_name, model, test_df):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} (2-class)")
    print(f"{'='*60}")

    test_dataset = RoadDataset(test_df, transform=test_transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=64,
        shuffle=False, num_workers=4, pin_memory=False)

    print(f"Test tiles: {len(test_dataset):,}")

    overall_acc   = MetricAccumulator(num_classes=3)
    muni_acc      = {}
    landscape_acc = {}

    with torch.no_grad():
        for batch_idx, (images, masks, municipalities, landscapes) in \
                enumerate(test_loader):
            images  = images.cuda()
            masks   = masks.cuda()
            outputs = model(images)
            preds   = outputs.argmax(dim=1)

            overall_acc.update(preds, masks)

            for i in range(len(images)):
                muni = municipalities[i]
                land = landscapes[i]

                if muni not in muni_acc:
                    muni_acc[muni] = MetricAccumulator(num_classes=3)
                if land not in landscape_acc:
                    landscape_acc[land] = MetricAccumulator(num_classes=3)

                muni_acc[muni].update(preds[i:i+1], masks[i:i+1])
                landscape_acc[land].update(preds[i:i+1], masks[i:i+1])

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}", flush=True)

    # Overall results
    overall = overall_acc.compute()
    print(f"\n--- Overall Test Results ---")
    print(f"Road Macro F1:   {overall['road_macro_f1']:.4f}")
    print(f"Road mIoU:       {overall['road_miou']:.4f}")
    print(f"mIoU:            {overall['miou']:.4f}")
    print(f"F1 Primary:      {overall['f1_class_1']:.4f}")
    print(f"F1 Secondary:    {overall['f1_class_2']:.4f}")
    print(f"Prec Primary:    {overall['precision_class_1']:.4f}")
    print(f"Rec  Primary:    {overall['recall_class_1']:.4f}")
    print(f"Prec Secondary:  {overall['precision_class_2']:.4f}")
    print(f"Rec  Secondary:  {overall['recall_class_2']:.4f}")

    # Per municipality
    print(f"\n--- Per Municipality (micro-averaged) ---")
    muni_rows = []
    for muni in sorted(muni_acc.keys()):
        avg = muni_acc[muni].compute()
        print(f"  {muni:25s} F1={avg['road_macro_f1']:.4f} "
              f"Primary={avg['f1_class_1']:.4f} "
              f"Secondary={avg['f1_class_2']:.4f}")
        muni_rows.append({'municipality': muni, 'model': model_name, **avg})

    # Per landscape
    print(f"\n--- Per Landscape Type (micro-averaged) ---")
    land_rows = []
    for land in sorted(landscape_acc.keys()):
        avg = landscape_acc[land].compute()
        print(f"  {land:20s} F1={avg['road_macro_f1']:.4f} "
              f"Primary={avg['f1_class_1']:.4f} "
              f"Secondary={avg['f1_class_2']:.4f}")
        land_rows.append({'landscape_type': land, 'model': model_name, **avg})

    # Save CSVs
    pd.DataFrame([{'model': model_name, **overall}]).to_csv(
        METRICS_DIR / f"{model_name}_test_overall.csv", index=False)
    pd.DataFrame(muni_rows).to_csv(
        METRICS_DIR / f"{model_name}_test_per_municipality.csv", index=False)
    pd.DataFrame(land_rows).to_csv(
        METRICS_DIR / f"{model_name}_test_per_landscape.csv", index=False)

    return overall, muni_rows, land_rows


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
        img        = np.load(best_tile['image_path'])
        mask       = np.load(best_tile['mask_path'])

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
        Patch(facecolor='#F0F0F0', edgecolor='grey', label='Background'),
        Patch(facecolor='#FF0000', label='Primary roads (major+local)'),
        Patch(facecolor='#00C800', label='Secondary roads (minor)')
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('2-Class Test Predictions — All Models',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = FIGURES_DIR / 'test_predictions_all_models_2class.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tile_df  = pd.read_csv(TILE_INDEX)
    test_df  = tile_df[tile_df['split'] == 'test'].reset_index(drop=True)
    print(f"Test tiles:          {len(test_df):,}")
    print(f"Test municipalities: {test_df['municipality'].unique().tolist()}")

    model_names = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']

    print("\nLoading models...")
    models_dict = {}
    for name in model_names:
        model, epoch = load_model(name)
        models_dict[name] = model
        print(f"  {name} loaded (best epoch {epoch})")

    all_overall   = {}
    all_muni      = []
    all_landscape = []

    for model_name in model_names:
        overall, muni_rows, land_rows = evaluate_model(
            model_name, models_dict[model_name], test_df)
        all_overall[model_name] = overall
        all_muni.extend(muni_rows)
        all_landscape.extend(land_rows)
        torch.cuda.empty_cache()

    generate_visual(models_dict, test_df)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS — 2-CLASS — MICRO AVERAGED")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Road F1':>10} {'Primary F1':>12} "
          f"{'Secondary F1':>14} {'mIoU':>10}")
    print("-"*68)
    for name, res in all_overall.items():
        print(f"{name:<20} {res['road_macro_f1']:>10.4f} "
              f"{res['f1_class_1']:>12.4f} "
              f"{res['f1_class_2']:>14.4f} "
              f"{res['miou']:>10.4f}")

    # Save combined results
    combined = [{'model': n, **r} for n, r in all_overall.items()]
    pd.DataFrame(combined).to_csv(
        METRICS_DIR / 'test_results_all_models_2class.csv', index=False)
    pd.DataFrame(all_muni).to_csv(
        METRICS_DIR / 'test_results_per_municipality_2class.csv', index=False)
    pd.DataFrame(all_landscape).to_csv(
        METRICS_DIR / 'test_results_per_landscape_2class.csv', index=False)

    print(f"\nAll results saved to: {METRICS_DIR}")
    print(f"Figures saved to:     {FIGURES_DIR}")