# -*- coding: utf-8 -*-
"""
Two parts: (A) Audit of the discrepancy between 17/18 and 12/16; 
(B) Downgrading Fig 1C to the tumour level (pseudobulk).
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon, mannwhitneyu

# ==================================================================
# A. Why did 17/18 become 12/16? — Must audit first before deciding which to use
# ==================================================================
def audit(old, new, key="tumour"):
    """
    old/new: DataFrame, columns = [key, delta], representing the per-tumour Δ from two pipeline versions.
    Output: Flipped tumours, amplitude changes, and the source of the n difference.
    """
    m = old.merge(new, on=key, suffixes=("_old", "_new"), how="outer")
    m["flipped"] = np.sign(m["delta_old"]) != np.sign(m["delta_new"])
    m["in_old_only"] = m["delta_new"].isna()
    m["in_new_only"] = m["delta_old"].isna()

    print("="*70); print("A  Discrepancy Audit"); print("="*70)
    print(f"  Old n={m['delta_old'].notna().sum()}  New n={m['delta_new'].notna().sum()}")
    print(f"  Only in old: {list(m.loc[m.in_old_only, key])}")
    print(f"  Only in new: {list(m.loc[m.in_new_only, key])}")
    print(f"  Tumours with flipped signs ({int(m['flipped'].fillna(False).sum())}):")
    print(m.loc[m["flipped"].fillna(False), [key,"delta_old","delta_new"]].to_string(index=False))
    
    print("\n  Potential causes (check one by one):")
    for t in [
        "1) Different PAN/LE definitions: signature argmax vs region_label vs distance to necrosis",
        "2) Different aggregation levels: section-level averaging vs direct tumour-level pooling",
        "3) Whether depth correction was applied (EV ~ n_counts residuals)",
        "4) Whether IDH-mut / recurrence samples were included",
        "5) Whether minimum spot counts were required (e.g., n >= 20)",
        "6) Different EV score versions (89-gene vs 23 core genes)",
    ]:
        print("   ", t)
    return m

# ==================================================================
# B. Downgrading Fig 1C to tumour level: pseudobulk per patient × cluster
# ==================================================================
def fig1c_tumour_level(meta, ev="EV_score", clus="seurat_clusters",
                       patient="patient", target=(10, 11), min_cells=20):
    """
    Correct approach: one mean per (patient, cluster/group) -> paired comparison within patients.
    Prohibited: Direct Mann-Whitney on hundreds of thousands of cells.
    """
    m = meta.copy()
    m["grp"] = np.where(m[clus].isin(target), "Target", "Other")

    # pseudobulk: patient × group
    pb = (m.groupby([patient, "grp"])[ev]
            .agg(["mean", "count"]).reset_index())
    pb = pb[pb["count"] >= min_cells]
    w = pb.pivot(index=patient, columns="grp", values="mean").dropna()
    
    if "Target" not in w or "Other" not in w:
        print("  Missing groups, check target cluster numbers"); return None

    d = w["Target"] - w["Other"]
    k, n = int((d > 0).sum()), len(d)

    print("\n"+"="*70); print("B  Fig 1C Tumour-level (pseudobulk)"); print("="*70)
    print(f"  Available patients n = {n} (>= {min_cells} cells per patient/group)")
    print(f"  Target > Other in: {k}/{n} patients")
    print(f"  Median Δ = {np.median(d):+.4f}   IQR [{np.percentile(d,25):+.4f}, {np.percentile(d,75):+.4f}]")
    print(f"  Sign test       p = {binomtest(k,n,0.5).pvalue:.5f}")
    print(f"  Wilcoxon        p = {wilcoxon(d).pvalue:.5f}")

    # Control: cell-level (demonstrating why it cannot be used for inference)
    a = m.loc[m["grp"] == "Target", ev]; b = m.loc[m["grp"] == "Other", ev]
    print(f"\n  [Control] Cell-level Mann-Whitney: n={len(a)} vs {len(b)}, "
          f"p={mannwhitneyu(a,b).pvalue:.2e}")
    print(f"         Cell-level median diff = {np.median(a)-np.median(b):+.4f}  <- Minor diff yields extremely small p")
    print(f"  => The 'all p < 0.001' in the manuscript should be replaced by these tumour-level results.")

    # leave-one-out
    loo = []
    for i in range(len(d)):
        dd = np.delete(d.values, i)
        loo.append(wilcoxon(dd).pvalue)
    print(f"\n  leave-one-out: Wilcoxon p range [{min(loo):.4f}, {max(loo):.4f}]")
    print(f"     {'Robust' if max(loo)<0.05 else 'Fragile: driven by a single influential observation'}")
    return d

# ==================================================================
# C. Robustness diagnostics for 12/16 Wilcoxon (why 0.013 cannot be the primary result)
# ==================================================================
def diagnose_12of16(d):
    d = np.asarray(d, float)
    k, n = int((d>0).sum()), len(d)
    print("\n"+"="*70); print("C  12/16 Robustness Diagnostics"); print("="*70)
    print(f"  Concordant {k}/{n} | median Δ={np.median(d):+.4f}")
    print(f"  Sign test p={binomtest(k,n,0.5).pvalue:.4f}   <- Pre-specified primary test in Methods")
    print(f"  Wilcoxon  p={wilcoxon(d).pvalue:.4f}   <- Parallel sensitivity analysis")
    print(f"  Median positive Δ={np.median(d[d>0]):+.4f}  Median negative Δ={np.median(d[d<0]):+.4f}")
    
    print("\n  Discrepancy indicates: significance is driven by a few extreme tumours, lacking directional consistency.")
    print("  Correct reporting: report both, using the pre-specified sign test as primary, and Wilcoxon as sensitivity.")
    
    loo=[wilcoxon(np.delete(d,i)).pvalue for i in range(n)]
    print(f"\n  Wilcoxon leave-one-out p range [{min(loo):.4f}, {max(loo):.4f}]")
    print(f"     {'Robust' if max(loo)<0.05 else 'WARNING: Fragile, loses significance upon removing any single tumour'}")
    return d

if __name__ == "__main__":
    # ---- Replace the following lines with your real data ----
    # meta = pd.read_csv("gbm_full_metadata.csv")
    # d_old = pd.read_csv("delta_old.csv"); d_new = pd.read_csv("delta_new.csv")
    # audit(d_old, d_new)
    # fig1c_tumour_level(meta)
    print("Please uncomment and replace with real data to run.")