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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# Paths
TILE_INDEX  = ROOT_DIR / "data/processed/metadata/tile_index_road_only.csv"
MODELS_DIR  = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "outputs/metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'tile_size':     512,
    'num_classes':   4,
    'batch_size':    64,
    'num_epochs':    50,
    'patience':      10,
    'learning_rate': 1e-4,
    'num_workers':   4,
    'random_seed':   42,
    'class_weights': [1.0, 8.0, 6.0, 4.0]
}

torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

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
        return image, mask.long()

class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32).cuda()
        self.ce_loss   = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    
    def forward(self, predictions, targets):
        return 0.5 * self.ce_loss(predictions, targets) + \
               0.5 * self.dice_loss(predictions, targets)

def calculate_metrics(preds, targets, num_classes=4):
    preds = preds.argmax(dim=1)
    metrics = {}
    iou_per_class = []
    f1_per_class  = []
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
        iou_per_class.append(iou)
        f1_per_class.append(f1)
        metrics[f'iou_class_{cls}'] = iou
        metrics[f'f1_class_{cls}']  = f1
    metrics['miou']      = np.mean(iou_per_class)
    metrics['macro_f1']  = np.mean(f1_per_class)
    metrics['road_miou'] = np.mean(iou_per_class[1:])
    return metrics

def get_model(model_name):
    if model_name == 'unet_resnet50':
        return smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3, classes=4)
    elif model_name == 'dlinknet':
        return smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3, classes=4,
            decoder_use_batchnorm=True)
    elif model_name == 'segformer':
        return smp.Segformer(
            encoder_name='mit_b2',
            encoder_weights='imagenet',
            in_channels=3, classes=4)
    elif model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3, classes=4)

def train_model(model_name, model, train_loader, val_loader, config):
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-6)
    criterion = CombinedLoss(config['class_weights'])
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{model_name}_best.pth"
    last_path = model_dir / f"{model_name}_last.pth"
    best_miou      = 0.0
    patience_count = 0
    history        = []
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        t_epoch = time.time()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.cuda()
            masks  = masks.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}", flush=True)
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss    = 0.0
        all_metrics = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks  = masks.cuda()
                outputs = model(images)
                loss    = criterion(outputs, masks)
                val_loss += loss.item()
                all_metrics.append(calculate_metrics(outputs, masks))
        avg_val_loss = val_loss / len(val_loader)
        epoch_metrics = {}
        for key in all_metrics[0].keys():
            epoch_metrics[key] = np.mean([m[key] for m in all_metrics])
        val_miou    = epoch_metrics['road_miou']
        epoch_time  = time.time() - t_epoch
        print(f"\nEpoch {epoch:02d}/{config['num_epochs']} | "
              f"Time: {epoch_time:.0f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Road mIoU: {val_miou:.4f} | "
              f"mIoU: {epoch_metrics['miou']:.4f}", flush=True)
        print(f"  Class IoU — Major: {epoch_metrics['iou_class_1']:.4f} | "
              f"Local: {epoch_metrics['iou_class_2']:.4f} | "
              f"Minor: {epoch_metrics['iou_class_3']:.4f}", flush=True)
        history.append({
            'epoch': epoch, 'train_loss': avg_train_loss,
            'val_loss': avg_val_loss, **epoch_metrics})
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_miou': val_miou, 'config': config
        }, last_path)
        if val_miou > best_miou:
            best_miou      = val_miou
            patience_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': val_miou, 'config': config
            }, best_path)
            print(f"  *** New best model saved — Road mIoU: {best_miou:.4f} ***", flush=True)
        else:
            patience_count += 1
            print(f"  No improvement — patience: {patience_count}/{config['patience']}", flush=True)
        if patience_count >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}", flush=True)
            break
    pd.DataFrame(history).to_csv(
        METRICS_DIR / f"{model_name}_history.csv", index=False)
    print(f"\nTraining complete — Best Road mIoU: {best_miou:.4f}", flush=True)
    return best_miou

if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    tile_df  = pd.read_csv(TILE_INDEX)
    train_df = tile_df[tile_df['split'] == 'train'].reset_index(drop=True)
    val_df   = tile_df[tile_df['split'] == 'val'].reset_index(drop=True)
    
    train_dataset = RoadDataset(train_df, transform=train_transform)
    val_dataset   = RoadDataset(val_df,   transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=False)
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")
    
    # Train models one by one
    models_to_train = [
        'unet_resnet50',
        'dlinknet',
        'segformer',
        'deeplabv3plus'
    ]
    
    results = {}
    for model_name in models_to_train:
        print(f"\nStarting {model_name}...")
        model = get_model(model_name)
        best_miou = train_model(
            model_name, model,
            train_loader, val_loader, CONFIG)
        results[model_name] = best_miou
        torch.cuda.empty_cache()
    
    print("\n=== FINAL RESULTS ===")
    for name, miou in results.items():
        print(f"{name}: {miou:.4f}")
        