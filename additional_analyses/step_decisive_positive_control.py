# -*- coding: utf-8 -*-
"""
Decisive Positive Control + Continuous State Axis Analysis

Must be run before declaring any negative conclusions. 
Without a positive control, "no difference" is uninterpretable.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr, mannwhitneyu

rng = np.random.default_rng(0)
df = pd.read_csv("gbm_full_metadata.csv")
CT, MAL = "cell_type", "Malignant"      # Modify according to your PART 0 output

# ---------- Gene Sets ----------
HYPOXIA = ["CA9","VEGFA","SLC2A1","PGK1","LDHA","ALDOA","ENO1","ADM","NDRG1","P4HA1",
           "PLOD2","BNIP3","DDIT4","ANKRD37","PDK1","HK2","PFKFB3","PFKFB4","EGLN3","HILPDA"]
MES = ["CD44","CHI3L1","CHI3L2","MET","TIMP1","FN1","COL1A1","SERPINE1","ANXA2","VIM",
       "LGALS1","S100A10","EMP3","TGFBI","TAGLN","ACTA2","CRYAB","HSPA1A","HSPA1B","DNAJB1",
       "HSPB1","JUN","JUNB","FOS","EGR1","ATF3","SOD2","TXN","GPX1","HMOX1","SLC2A1","PGK1"]
# Negative control gene set: random housekeeping genes (should be completely independent of regions)
HOUSE  = ["ACTB","GAPDH","TUBB","PSAP","SRP14","RPL13A","RPS18","EEF1A1","TPT1","B2M"]

def zscore_set(sub, genes, name):
    g = [x for x in genes if x in sub.columns]
    if len(g) < 5:
        print(f"  [FAIL] {name}: Only {len(g)} genes available -> Requires raw expression matrix (metadata lacks gene columns)")
        return None
    z = sub[g].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    sub[name] = z.mean(axis=1)
    print(f"  [OK] {name} ({len(g)} genes)")
    return name

def paired(sub, val):
    m = sub.groupby(["patient","region"])[val].mean().unstack()[["tmc","para"]].dropna()
    d = m["tmc"] - m["para"]
    k, n = int((d > 0).sum()), len(d)
    lo, hi = np.percentile([np.median(rng.choice(d, len(d), True)) for _ in range(5000)], [2.5, 97.5])
    print(f"    {k}/{n} concordant | Median Delta = {np.median(d):+.4f} CI [{lo:+.4f}, {hi:+.4f}] "
          f"| sign p = {binomtest(k, n, 0.5).pvalue:.5f} | Wilcoxon p = {wilcoxon(d).pvalue:.5f}")
    return d

print("="*72)
print("STEP 1  Positive Control (Determines if the negative conclusion is valid)")
print("="*72)
print("Running the same pipeline on Hallmark Hypoxia / MES / Housekeeping genes:")
ok = {}
for nm, gs in [("Hypoxia", HYPOXIA), ("MES", MES), ("Housekeeping", HOUSE)]:
    if zscore_set(df, gs, nm):
        print(f"  [{nm}] tmc vs para (n=16 paired):")
        ok[nm] = paired(df, nm)

print("\n  Interpretation (This step is critical):")
if "Hypoxia" in ok:
    d = ok["Hypoxia"]
    k = int((d > 0).sum())
    if k >= 14 and np.median(d) > 0.05:
        print("   [PASS] Hypoxia is strongly positive -> Pipeline and labels are sensitive.")
        print("          EV is significant in all cells but not in malignant cells -> Credible negative conclusion.")
    elif k >= 12:
        print("   [WARN] Hypoxia is weakly positive -> Pipeline is marginally acceptable, negative conclusion requires cautious phrasing.")
    else:
        print("   [FAIL] Hypoxia is also non-significant -> Pipeline or region labels are flawed.")
        print("          All conclusions (including negative ones) are uninterpretable. Fix pipeline, do not submit.")

print("\n" + "="*72)
print("STEP 2  Continuous State Axis (Inter-cluster equivalence also cannot disprove it)")
print("="*72)
mal = df[df[CT].eq(MAL)]
for nm, gs in [("Hypoxia", HYPOXIA), ("MES", MES)]:
    if nm not in mal.columns: zscore_set(mal, gs, nm)
    if nm in mal.columns:
        print(f"\n  [{nm} Axis, within malignant cells]")
        # A. Tumour-level correlation (anti-pseudoreplication): one value per tumour
        pt = mal.groupby("patient").apply(lambda s: pd.Series({"ev": s["EV_score"].mean(), "st": s[nm].mean()}))
        rr = spearmanr(pt["ev"], pt["st"])
        print(f"    A) Tumour-level EV vs {nm}: rho={rr.statistic:+.3f}, p={rr.pvalue:.4f}, n={len(pt)}")
        # B. Within-tumour quartile comparison (highest power, independent of region labels)
        qs = []
        for pid, s in mal.groupby("patient"):
            if len(s) < 40: continue
            hiq = s[s[nm] >= s[nm].quantile(0.75)]["EV_score"].mean()
            loq = s[s[nm] <= s[nm].quantile(0.25)]["EV_score"].mean()
            qs.append(hiq - loq)
        qs = np.array(qs)
        kk = int((qs > 0).sum())
        print(f"    B) Within-tumour High-{nm} minus Low-{nm} quartile EV diff: {kk}/{len(qs)} concordant, "
              f"median={np.median(qs):+.4f}, sign p={binomtest(kk, len(qs), 0.5).pvalue:.5f}")
        if binomtest(kk, len(qs), 0.5).pvalue < 0.05:
            print("       -> [PASS] Hypoxia chain is established at the state level")
        else:
            print("       -> [FAIL] Hypoxia chain is not closed")

print("\n" + "="*72)
print("STEP 3  Pseudoreplication Warning (Optional, methodological bonus)")
print("="*72)
if "seurat_clusters" in df.columns:
    cl = df["seurat_clusters"].value_counts()
    print(f"  13 clusters, median {int(cl.median())} cells per cluster")
    a = df[df["seurat_clusters"] == cl.index[0]]["EV_score"]
    b = df[df["seurat_clusters"] == cl.index[1]]["EV_score"]
    print(f"  Cluster {cl.index[0]} vs Cluster {cl.index[1]} cell-level Mann-Whitney p={mannwhitneyu(a, b).pvalue:.2e}")
    print("  Even if the median difference is only 0.02 (biologically meaningless), cell-level p remains extremely small -> Pseudoreplication example")

print("\n" + "="*72)
print("Summary of Interpretation Order:")
print("  1) STEP 1 Hypoxia not significant -> Stop, fix pipeline.")
print("  2) STEP 1 significant + STEP 2 significant -> Hypoxia chain established, paper is salvageable (at state level).")
print("  3) STEP 1 significant + STEP 2 not significant -> Credible negative conclusion, pursue methodological warning / negative result route.")