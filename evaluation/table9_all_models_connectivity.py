"""
table9_all_models_connectivity.py
Computes correctness, completeness, and quality for all 4 models
on all 5 test municipalities, both 3-class and 2-class schemes.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

TILE_INDEX_3  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_road_only.csv")
TILE_INDEX_2  = Path(r"D:\lstojano\road_extraction_slovenia\data\processed\metadata\tile_index_2class.csv")
MODELS_3CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models")
MODELS_2CLASS = Path(r"D:\lstojano\road_extraction_slovenia\models\2class")
OUT_DIR       = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES  = ['unet_resnet50', 'dlinknet', 'segformer', 'deeplabv3plus']
MODEL_LABELS = {
    'unet_resnet50':  'U-Net/ResNet50',
    'dlinknet':       'D-LinkNet',
    'segformer':      'SegFormer-B2',
    'deeplabv3plus':  'DeepLabV3+'
}

test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

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
    model = get_model(model_name, num_classes).cuda()
    ckpt  = torch.load(
        model_dir / model_name / f"{model_name}_best.pth",
        map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def predict_single(model, img):
    aug    = test_transform(image=img,
                            mask=np.zeros(img.shape[:2], dtype=np.uint8))
    tensor = aug['image'].unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(tensor)
    return out.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

def compute_municipality_metrics(tile_rows, model):
    tp_total = fp_total = fn_total = 0
    for idx, (_, tile) in enumerate(tile_rows.iterrows()):
        img  = np.load(tile['image_path'])
        ref  = np.load(tile['mask_path'])
        pred = predict_single(model, img)

        pred_road = (pred > 0)
        ref_road  = (ref  > 0)

        tp_total += int(( pred_road &  ref_road).sum())
        fp_total += int(( pred_road & ~ref_road).sum())
        fn_total += int((~pred_road &  ref_road).sum())

        if (idx + 1) % 2000 == 0:
            print(f"      {idx+1}/{len(tile_rows)} tiles...",
                  flush=True)

    correctness  = tp_total / (tp_total + fp_total + 1e-6)
    completeness = tp_total / (tp_total + fn_total + 1e-6)
    quality      = tp_total / (tp_total + fp_total + fn_total + 1e-6)

    return {
        'correctness':  round(float(correctness),  4),
        'completeness': round(float(completeness), 4),
        'quality':      round(float(quality),      4),
    }


if __name__ == '__main__':
    print("="*70)
    print("TABLE 9 — ALL-MODEL CONNECTIVITY METRICS")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tile_df_3 = pd.read_csv(TILE_INDEX_3)
    tile_df_2 = pd.read_csv(TILE_INDEX_2)
    test_df_3 = tile_df_3[tile_df_3['split'] == 'test'].reset_index(drop=True)
    test_df_2 = tile_df_2[tile_df_2['split'] == 'test'].reset_index(drop=True)

    municipalities = sorted(test_df_3['municipality'].unique())
    landscape_map  = dict(zip(test_df_3['municipality'],
                               test_df_3['landscape_type']))

    all_records = []

    for scheme, model_dir, test_df, num_classes in [
            ('3class', MODELS_3CLASS, test_df_3, 4),
            ('2class', MODELS_2CLASS, test_df_2, 3)]:

        print(f"\n{'='*60}")
        print(f"Scheme: {scheme}")
        print(f"{'='*60}")

        for model_name in MODEL_NAMES:
            print(f"\n  {MODEL_LABELS[model_name]}...")
            model = load_model(model_name, model_dir, num_classes)

            for muni in municipalities:
                muni_tiles = test_df[test_df['municipality'] == muni]
                landscape  = landscape_map.get(muni, '')
                print(f"    {muni} ({len(muni_tiles):,} tiles)...",
                      flush=True)

                metrics = compute_municipality_metrics(muni_tiles, model)

                print(f"      Cor={metrics['correctness']:.4f} | "
                      f"Com={metrics['completeness']:.4f} | "
                      f"Q={metrics['quality']:.4f}")

                all_records.append({
                    'scheme':        scheme,
                    'model':         MODEL_LABELS[model_name],
                    'municipality':  muni,
                    'landscape':     landscape,
                    **metrics
                })

            del model
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(OUT_DIR / 'table9_all_models_connectivity.csv',
                      index=False)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for scheme in ['3class', '2class']:
        sdf = results_df[results_df['scheme'] == scheme]
        print(f"\n--- {scheme} ---")
        print(f"{'Model':22s} {'Municipality':20s} {'Landscape':15s} "
              f"{'Correct.':>10} {'Complete.':>10} {'Quality':>10}")
        print("-"*90)
        for model_name in MODEL_NAMES:
            label = MODEL_LABELS[model_name]
            mdf   = sdf[sdf['model'] == label]
            for _, row in mdf.iterrows():
                print(f"{label:22s} {row['municipality']:20s} "
                      f"{row['landscape']:15s} "
                      f"{row['correctness']:>10.4f} "
                      f"{row['completeness']:>10.4f} "
                      f"{row['quality']:>10.4f}")
            # Mean row
            print(f"{'  → mean':22s} {'':20s} {'':15s} "
                  f"{mdf['correctness'].mean():>10.4f} "
                  f"{mdf['completeness'].mean():>10.4f} "
                  f"{mdf['quality'].mean():>10.4f}")
            print()

    print(f"\nSaved: {OUT_DIR}/table9_all_models_connectivity.csv")