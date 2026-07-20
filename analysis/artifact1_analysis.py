"""
artifact1_analysis.py
Statistical analysis of per-tile counts from artifact1_per_tile_counts.csv.
Computes:
  1. Bootstrap confidence intervals per model per scheme
  2. Paired bootstrap test for model comparisons
  3. Per-landscape uncertainty (within-municipality F1 distribution)
Run AFTER artifact1_per_tile_counts.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR    = Path(r"D:\lstojano\road_extraction_slovenia\outputs\article")
COUNTS_CSV = OUT_DIR / 'artifact1_per_tile_counts.csv'

N_BOOTSTRAP = 10000
ALPHA       = 0.05   # 95% confidence intervals
np.random.seed(42)

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading per-tile counts...")
df = pd.read_csv(COUNTS_CSV)
print(f"  {len(df):,} rows | "
      f"Models: {df['model'].unique().tolist()} | "
      f"Schemes: {df['scheme'].unique().tolist()}")

# ── Helper — compute Road Macro F1 from aggregated counts ─────────────────
def road_macro_f1_from_counts(sub_df, scheme):
    if scheme == '3class':
        classes = ['major', 'local', 'minor']
    else:
        classes = ['primary', 'secondary']

    f1s = []
    for cls in classes:
        tp_col = f'tp_{cls}'
        fp_col = f'fp_{cls}'
        fn_col = f'fn_{cls}'
        if tp_col not in sub_df.columns:
            continue
        tp   = sub_df[tp_col].sum()
        fp   = sub_df[fp_col].sum()
        fn   = sub_df[fn_col].sum()
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        f1s.append(float(f1))
    return float(np.mean(f1s)) if f1s else 0.0

def bootstrap_ci(tile_ids, sub_df, scheme, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """Bootstrap CI for Road Macro F1 by resampling tiles."""
    unique_tiles = tile_ids.unique()
    n            = len(unique_tiles)
    boot_f1s     = []

    for _ in range(n_boot):
        sampled = np.random.choice(unique_tiles, size=n, replace=True)
        boot_df = sub_df[sub_df['tile_id'].isin(sampled)]
        f1      = road_macro_f1_from_counts(boot_df, scheme)
        boot_f1s.append(f1)

    boot_f1s = np.array(boot_f1s)
    return {
        'mean':  float(np.mean(boot_f1s)),
        'std':   float(np.std(boot_f1s)),
        'ci_lo': float(np.percentile(boot_f1s, 100 * alpha / 2)),
        'ci_hi': float(np.percentile(boot_f1s, 100 * (1 - alpha / 2))),
    }

def paired_bootstrap(tile_ids, df_a, df_b, scheme,
                     n_boot=N_BOOTSTRAP, alpha=ALPHA):
    """
    Paired bootstrap test: is model A significantly better than model B?
    Returns CI of (F1_A - F1_B) and p-value (proportion of boot where A<=B).
    """
    unique_tiles = tile_ids.unique()
    n            = len(unique_tiles)
    diffs        = []

    for _ in range(n_boot):
        sampled  = np.random.choice(unique_tiles, size=n, replace=True)
        boot_a   = df_a[df_a['tile_id'].isin(sampled)]
        boot_b   = df_b[df_b['tile_id'].isin(sampled)]
        f1_a     = road_macro_f1_from_counts(boot_a, scheme)
        f1_b     = road_macro_f1_from_counts(boot_b, scheme)
        diffs.append(f1_a - f1_b)

    diffs  = np.array(diffs)
    p_val  = float((diffs <= 0).mean())  # one-sided: P(A not > B)
    return {
        'mean_diff': float(np.mean(diffs)),
        'ci_lo':     float(np.percentile(diffs, 100 * alpha / 2)),
        'ci_hi':     float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        'p_value':   p_val,
        'significant': p_val < alpha
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Bootstrap CIs per model
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SECTION 1 — Bootstrap Confidence Intervals ({N_BOOTSTRAP:,} samples)")
print(f"{'='*70}")

ci_records = []

for scheme in ['3class', '2class']:
    sdf = df[df['scheme'] == scheme]
    print(f"\n  Scheme: {scheme}")
    for model in df['model'].unique():
        mdf = sdf[sdf['model'] == model]
        obs = road_macro_f1_from_counts(mdf, scheme)
        ci  = bootstrap_ci(mdf['tile_id'], mdf, scheme)
        print(f"    {model:20s} F1={obs:.4f} | "
              f"95% CI [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] | "
              f"std={ci['std']:.4f}")
        ci_records.append({
            'model':    model,
            'scheme':   scheme,
            'observed_f1': obs,
            **ci
        })

ci_df = pd.DataFrame(ci_records)
ci_df.to_csv(OUT_DIR / 'artifact1_bootstrap_ci.csv', index=False)
print(f"\n  Saved: artifact1_bootstrap_ci.csv")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Paired model comparisons
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SECTION 2 — Paired Bootstrap Model Comparisons")
print(f"{'='*70}")

model_names = df['model'].unique().tolist()
paired_records = []

for scheme in ['3class', '2class']:
    sdf = df[df['scheme'] == scheme]
    print(f"\n  Scheme: {scheme}")
    for i, m_a in enumerate(model_names):
        for m_b in model_names[i+1:]:
            df_a = sdf[sdf['model'] == m_a]
            df_b = sdf[sdf['model'] == m_b]

            # Use intersection of tile_ids for fair comparison
            common = set(df_a['tile_id']) & set(df_b['tile_id'])
            df_a_c = df_a[df_a['tile_id'].isin(common)]
            df_b_c = df_b[df_b['tile_id'].isin(common)]

            result = paired_bootstrap(
                df_a_c['tile_id'], df_a_c, df_b_c, scheme)

            sig_str = '*** SIGNIFICANT' if result['significant'] else ''
            print(f"    {m_a:20s} vs {m_b:20s} | "
                  f"Δ={result['mean_diff']:+.4f} "
                  f"[{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}] | "
                  f"p={result['p_value']:.4f} {sig_str}")

            paired_records.append({
                'model_a':   m_a,
                'model_b':   m_b,
                'scheme':    scheme,
                **result
            })

paired_df = pd.DataFrame(paired_records)
paired_df.to_csv(OUT_DIR / 'artifact1_paired_bootstrap.csv', index=False)
print(f"\n  Saved: artifact1_paired_bootstrap.csv")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Per-landscape uncertainty
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SECTION 3 — Per-Landscape F1 Distribution")
print(f"{'='*70}")

landscape_records = []

scheme = '3class'
sdf    = df[df['scheme'] == scheme]

for landscape in sdf['landscape'].unique():
    ldf = sdf[sdf['landscape'] == landscape]
    for model in model_names:
        mdf = ldf[ldf['model'] == model]
        if len(mdf) == 0:
            continue

        # Per-tile F1 scores
        tile_f1s = []
        for tile_id, tdf in mdf.groupby('tile_id'):
            f1 = road_macro_f1_from_counts(tdf, scheme)
            tile_f1s.append(f1)

        tile_f1s = np.array(tile_f1s)
        overall  = road_macro_f1_from_counts(mdf, scheme)

        landscape_records.append({
            'landscape':       landscape,
            'model':           model,
            'n_tiles':         len(tile_f1s),
            'overall_f1':      round(overall, 4),
            'mean_tile_f1':    round(float(tile_f1s.mean()), 4),
            'std_tile_f1':     round(float(tile_f1s.std()),  4),
            'median_tile_f1':  round(float(np.median(tile_f1s)), 4),
            'q25_tile_f1':     round(float(np.percentile(tile_f1s, 25)), 4),
            'q75_tile_f1':     round(float(np.percentile(tile_f1s, 75)), 4),
            'min_tile_f1':     round(float(tile_f1s.min()), 4),
            'max_tile_f1':     round(float(tile_f1s.max()), 4),
        })

landscape_df = pd.DataFrame(landscape_records)
landscape_df.to_csv(OUT_DIR / 'artifact1_landscape_uncertainty.csv',
                    index=False)

print(f"\n  Per-landscape F1 distribution (SegFormer, 3-class):")
seg_df = landscape_df[landscape_df['model'] == 'segformer']
print(seg_df[[
    'landscape', 'n_tiles', 'overall_f1',
    'mean_tile_f1', 'std_tile_f1',
    'q25_tile_f1', 'q75_tile_f1'
]].to_string(index=False))
print(f"\n  Saved: artifact1_landscape_uncertainty.csv")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"ARTIFACT 1 ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"Files saved to: {OUT_DIR}")
print(f"  artifact1_bootstrap_ci.csv       — CIs per model")
print(f"  artifact1_paired_bootstrap.csv   — pairwise significance tests")
print(f"  artifact1_landscape_uncertainty.csv — within-landscape variance")