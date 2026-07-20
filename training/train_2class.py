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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# Paths
TILE_INDEX  = ROOT_DIR / "data/processed/metadata/tile_index_2class.csv"
MODELS_DIR  = ROOT_DIR / "models/2class"
METRICS_DIR = ROOT_DIR / "outputs/2class/metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'tile_size':     512,
    'num_classes':   3,        # background + primary + secondary
    'batch_size':    32,
    'num_epochs':    50,
    'patience':      10,
    'learning_rate': 1e-4,
    'num_workers':   4,
    'random_seed':   42,
    'class_weights': [1.0, 6.0, 4.0]  # bg, primary, secondary
}

# Full reproducibility
torch.manual_seed(CONFIG['random_seed'])
torch.cuda.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])
random.seed(CONFIG['random_seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
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
        return image, mask.long()

# Loss
class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        weights        = torch.tensor(class_weights,
                                      dtype=torch.float32).cuda()
        self.ce_loss   = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass', from_logits=True)

    def forward(self, predictions, targets):
        return 0.5 * self.ce_loss(predictions, targets) + \
               0.5 * self.dice_loss(predictions, targets)

# Metrics
def calculate_metrics(preds, targets, num_classes=3):
    preds    = preds.argmax(dim=1)
    road_f1  = []
    road_iou = []
    metrics  = {}

    for cls in range(num_classes):
        pred_cls     = (preds == cls)
        target_cls   = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()
        iou          = (intersection / (union + 1e-6)).item()
        tp           = intersection.item()
        fp           = (pred_cls & ~target_cls).sum().float().item()
        fn           = (~pred_cls & target_cls).sum().float().item()
        precision    = tp / (tp + fp + 1e-6)
        recall       = tp / (tp + fn + 1e-6)
        f1           = 2 * precision * recall / (precision + recall + 1e-6)

        metrics[f'iou_class_{cls}']       = iou
        metrics[f'f1_class_{cls}']        = f1
        metrics[f'precision_class_{cls}'] = precision
        metrics[f'recall_class_{cls}']    = recall

        if cls > 0:
            road_f1.append(f1)
            road_iou.append(iou)

    metrics['miou']           = np.mean(
        [metrics[f'iou_class_{c}'] for c in range(num_classes)])
    metrics['macro_f1']       = np.mean(
        [metrics[f'f1_class_{c}'] for c in range(num_classes)])
    metrics['road_macro_f1']  = np.mean(road_f1)
    metrics['road_miou']      = np.mean(road_iou)
    return metrics

# Models
def get_model(model_name, num_classes):
    if model_name == 'unet_resnet50':
        return smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3, classes=num_classes)
    elif model_name == 'dlinknet':
        return smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3, classes=num_classes,
            decoder_use_batchnorm=True)
    elif model_name == 'segformer':
        return smp.Segformer(
            encoder_name='mit_b2',
            encoder_weights='imagenet',
            in_channels=3, classes=num_classes)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3, classes=num_classes)

# Training loop
def train_model(model_name, model, train_loader, val_loader, config):
    model      = model.cuda()
    optimizer  = AdamW(model.parameters(),
                       lr=config['learning_rate'],
                       weight_decay=1e-4)
    scheduler  = CosineAnnealingLR(optimizer,
                                    T_max=config['num_epochs'],
                                    eta_min=1e-6)
    criterion  = CombinedLoss(config['class_weights'])

    model_dir  = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path  = model_dir / f"{model_name}_best.pth"
    last_path  = model_dir / f"{model_name}_last.pth"

    best_f1        = 0.0
    patience_count = 0
    history        = []

    print(f"\n{'='*60}")
    print(f"Training: {model_name} (2-class)")
    print(f"{'='*60}")

    for epoch in range(1, config['num_epochs'] + 1):

        # Training
        model.train()
        train_loss = 0.0
        t_epoch    = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.cuda()
            masks  = masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/"
                      f"{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss    = 0.0
        all_metrics = []

        with torch.no_grad():
            for images, masks in val_loader:
                images  = images.cuda()
                masks   = masks.cuda()
                outputs = model(images)
                loss    = criterion(outputs, masks)
                val_loss += loss.item()
                all_metrics.append(
                    calculate_metrics(outputs, masks,
                                      config['num_classes']))

        avg_val_loss  = val_loss / len(val_loader)
        epoch_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        val_f1     = epoch_metrics['road_macro_f1']
        epoch_time = time.time() - t_epoch

        print(f"\nEpoch {epoch:02d}/{config['num_epochs']} | "
              f"Time: {epoch_time:.0f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Road Macro F1: {val_f1:.4f} | "
              f"mIoU: {epoch_metrics['miou']:.4f}", flush=True)
        print(f"  Class F1  — Primary: {epoch_metrics['f1_class_1']:.4f} | "
              f"Secondary: {epoch_metrics['f1_class_2']:.4f}", flush=True)
        print(f"  Class IoU — Primary: {epoch_metrics['iou_class_1']:.4f} | "
              f"Secondary: {epoch_metrics['iou_class_2']:.4f}", flush=True)

        history.append({
            'epoch':      epoch,
            'train_loss': avg_train_loss,
            'val_loss':   avg_val_loss,
            **epoch_metrics
        })

        # Save last
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1':               val_f1,
            'config':               config
        }, last_path)

        # Save best
        if val_f1 > best_f1:
            best_f1        = val_f1
            patience_count = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1':               val_f1,
                'config':               config
            }, best_path)
            print(f"  *** New best model saved — "
                  f"Road Macro F1: {best_f1:.4f} ***", flush=True)
        else:
            patience_count += 1
            print(f"  No improvement — "
                  f"patience: {patience_count}/{config['patience']}",
                  flush=True)

        if patience_count >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}", flush=True)
            break

    pd.DataFrame(history).to_csv(
        METRICS_DIR / f"{model_name}_history.csv", index=False)
    print(f"\nTraining complete — Best Road Macro F1: {best_f1:.4f}",
          flush=True)
    return best_f1


if __name__ == '__main__':
    # Seeds inside main
    torch.manual_seed(CONFIG['random_seed'])
    torch.cuda.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    random.seed(CONFIG['random_seed'])

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {CONFIG}")

    # Load data
    tile_df  = pd.read_csv(TILE_INDEX)
    train_df = tile_df[tile_df['split'] == 'train'].reset_index(drop=True)
    val_df   = tile_df[tile_df['split'] == 'val'].reset_index(drop=True)

    print(f"\nTrain tiles: {len(train_df):,}")
    print(f"Val tiles:   {len(val_df):,}")

    # Dataloaders
    train_dataset = RoadDataset(train_df, transform=train_transform)
    val_dataset   = RoadDataset(val_df,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=False)

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    # Train all models
    models_to_train = [
        'unet_resnet50',
        'dlinknet',
        'segformer',
        'deeplabv3plus'
    ]

    results = {}
    for model_name in models_to_train:
        print(f"\nStarting {model_name}...")
        model   = get_model(model_name, CONFIG['num_classes'])
        best_f1 = train_model(
            model_name, model,
            train_loader, val_loader, CONFIG)
        results[model_name] = best_f1
        torch.cuda.empty_cache()

    print("\n=== FINAL RESULTS (2-class) ===")
    for name, f1 in results.items():
        print(f"{name}: Road Macro F1 = {f1:.4f}")