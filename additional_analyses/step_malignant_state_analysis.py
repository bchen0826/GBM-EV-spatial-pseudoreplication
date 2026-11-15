# -*- coding: utf-8 -*-
"""
Malignant cell state analysis (Neftel state / hypoxia) + Pre-filtering verification.
All comparisons are tumour-level paired. Cell-level p-values are strictly prohibited from entering conclusions.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr

df = pd.read_csv("gbm_full_metadata.csv")

# ---------- PART 0: Verify filtering (Essential) ----------
print("="*70); print("PART 0  Verification of filtering"); print("="*70)
print("Total cells:", len(df))
print("\nValues in candidate cell type columns:")
for col in df.columns:
    lc = col.lower()
    if any(k in lc for k in ["cell_type","celltype","cell.type","major","class","label","annot"]):
        vc = df[col].value_counts()
        if 2 <= len(vc) <= 30:
            print(f"\n  [{col}]  ({len(vc)} types)")
            print(vc.head(15).to_string())

CT = "cell_type"                    # Modify according to printed results
MAL = "Malignant"                   # Modify according to printed results

before = df.groupby(["patient","region"]).size().unstack()
mal_all = df[df[CT].eq(MAL)]
after  = mal_all.groupby(["patient","region"]).size().unstack()
print("\nCells per patient/region (before and after filtering):")
print(pd.concat([before.add_suffix("_all"), after.add_suffix("_mal")], axis=1).to_string())

print("\nMean EV score: Malignant vs Non-malignant:")
print(df.groupby(df[CT].eq(MAL))["EV_score"].mean().to_string())

def pair_delta(sub):
    m = sub.groupby(["patient","region"])["EV_score"].mean().unstack()
    m = m[["tmc","para"]].dropna()
    return m["tmc"] - m["para"]

d_all, d_mal = pair_delta(df), pair_delta(mal_all)
print("\nDelta comparison (if identical, filtering is ineffective or means are equal):")
print(pd.DataFrame({"delta_all": d_all, "delta_mal": d_mal}).to_string())
print(f"\n  All cells: {(d_all>0).sum()}/{len(d_all)}  Wilcoxon p={wilcoxon(d_all).pvalue:.6f}")
print(f"  Malignant: {(d_mal>0).sum()}/{len(d_mal)}  Wilcoxon p={wilcoxon(d_mal).pvalue:.6f}")

# ---------- PART 1: Depth correction (within malignant cells) ----------
print("\n"+"="*70); print("PART 1  Depth correction"); print("="*70)
if "nUMI" in df.columns or "nCount_RNA" in df.columns:
    depth_col = "nUMI" if "nUMI" in df.columns else "nCount_RNA"
    r = spearmanr(mal_all["EV_score"], mal_all[depth_col])
    print(f"EV score vs {depth_col}: Spearman rho = {r.statistic:.3f}, p = {r.pvalue:.3e}")
    if abs(r.statistic) > 0.3:
        print("  [WARN] Strong depth correlation, log(depth) must be added as a covariate in paired models.")
else:
    print("Depth column not found; manually specify total_counts / nFeature etc.")

# ---------- PART 2: Cell state / Hypoxia ----------
print("\n"+"="*70); print("PART 2  Cell state and Hypoxia"); print("="*70)
STATE_CANDS = ["Neftel_state","neftel","cell_state","state","Malignant_state",
               "subtype","Hypoxia_score","hypoxia","Hypoxia"]
found = [c for c in df.columns if c.lower() in [s.lower() for s in STATE_CANDS]]
print("Detected state/hypoxia columns:", found if found else "None -> Running marker scoring")

NEFTEL = {
 "MES-like": ["CD44","CHI3L1","CHI3L2","MET","TIMP1","FN1","COL1A1","SERPINE1","ANXA2","VIM","LGALS1","S100A10","EMP3","TGFBI","TAGLN","ACTA2","MT2A","CRYAB","HSPA1A","HSPA1B","DNAJB1","HSPB1","JUN","JUNB","FOS","EGR1","ZFP36","IER2","ATF3","SOD2","TXN","GPX1","HMOX1","SLC2A1","PGK1","LDHA","ALDOA","ENO1","CA9","VEGFA","ADM","NDRG1","P4HA1","PLOD2","BNIP3","DDIT4","TRIB3","CHAC1","ASNS","TRB3","SESN2","REDD1"],
 "AC-like":  ["GFAP","AQP4","SLC1A2","GJA1","SOX9","CLU","APOE","S100B","CD44","MLC1","AQP1","SPARCL1","TIMP3","ITIH2","PTN","NTRK2","SLC6A11","GLAST","FABP7","ALDH1L1","MLC1"],
 "OPC-like": ["PDGFRA","CSPG4","OLIG1","OLIG2","SOX10","NKX6-1","BCAN","VCAN","PTPRZ1","ASCL1","DLL3","EGFR","SOX4","TFAP2C","LIMS1","MAP2","GPR17","ST8SIA1","UGT8","PLP1","CNP","MOBP","MBP","MAG","MOG","CLDN11","GAL3ST1","TRIM67","APOD","PTGDS"],
 "NPC-like": ["SOX2","NES","FABP7","TOP2A","CCNB1","CDK1","BIRC5","MKI67","NUSAP1","TYMS","RRM2","MCM2","PCNA","SOX4","SOX11","DCX","TUBB3","STMN1","STMN2","HES6","ASCL1","DLL1","EOMES","HES5","PAX6","GLI1","MYCN","OLIG2","NG2"],
}
HYPOXIA = ["CA9","VEGFA","SLC2A1","PGK1","LDHA","ALDOA","ENO1","ADM","NDRG1","P4HA1",
           "PLOD2","BNIP3","DDIT4","ANKRD37","PDK1","HK2","PFKFB3","PFKFB4","GPI","TPI1",
           "GAPDH","TP11","EGLN3","FAM162B","SNHG12","HILPDA","NOL3","ARL4C","C15orf48"]

def score_genes(sub, genes, name):
    g = [x for x in genes if x in sub.columns]
    miss = len(genes) - len(g)
    if len(g) < 5:
        print(f"  [FAIL] {name}: Only {len(g)} genes available, skipping"); return None
    z = sub[g].apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    sub[name] = z.mean(axis=1)
    print(f"  [OK] {name}: {len(g)} genes ({miss} missing)")
    return name

if found:
    state_col = found[0]
    print(f"Using existing column: {state_col}")
    print(df[df[CT].eq(MAL)][state_col].value_counts().to_string())
else:
    print("\nNo state column found -> Scoring with Neftel markers (z-score mean)")
    for k, v in NEFTEL.items(): score_genes(mal_all, v, k)
    score_genes(mal_all, HYPOXIA, "Hypoxia_score")
    score_genes(mal_all, ["nUMI"] if "nUMI" in df.columns else [], "dummy_depth")
    # Assign discrete label based on max score per cell
    sc = [c for c in NEFTEL if c in mal_all.columns]
    if sc:
        mal_all["Neftel_state"] = mal_all[sc].idxmax(axis=1)
        print("\nState distribution after scoring:"); print(mal_all["Neftel_state"].value_counts().to_string())

# ---------- PART 3: Tumour-level paired comparison ----------
print("\n"+"="*70); print("PART 3  Tumour-level paired tests"); print("="*70)
def paired_test(sub, val, label):
    m = sub.groupby(["patient","region"])[val].mean().unstack()
    m = m[["tmc","para"]].dropna()
    d = m["tmc"] - m["para"]
    k, n = int((d>0).sum()), len(d)
    print(f"\n{label}: {k}/{n} concordant | sign p={binomtest(k,n,0.5).pvalue:.5f} "
          f"| Wilcoxon p={wilcoxon(d).pvalue:.5f} | median Δ={np.median(d):+.4f}")
    return d

if "Hypoxia_score" in mal_all.columns:
    paired_test(mal_all, "Hypoxia_score", "Malignant Hypoxia score (tmc vs para)")
    r = spearmanr(mal_all["EV_score"], mal_all["Hypoxia_score"])
    print(f"\nEV vs Hypoxia (cell-level): rho={r.statistic:.3f}, p={r.pvalue:.3e}")
    # Tumour-level correlation (anti-pseudoreplication)
    per_t = mal_all.groupby("patient").apply(
        lambda s: pd.Series({"ev":s["EV_score"].mean(),"hy":s["Hypoxia_score"].mean()}))
    rr = spearmanr(per_t["ev"], per_t["hy"])
    print(f"EV vs Hypoxia (tumour-level, n={len(per_t)}): rho={rr.statistic:.3f}, p={rr.pvalue:.4f}")

if "Neftel_state" in mal_all.columns:
    print("\ntmc vs para within each state (key: are all states concordant?):")
    for st, s in mal_all.groupby("Neftel_state"):
        if len(s) > 200: paired_test(s, "EV_score", f"  {st}")
    print("\nMean EV score by state (test if MES is highest):")
    print(mal_all.groupby("Neftel_state")["EV_score"].agg(["mean","count"]).sort_values("mean").to_string())

print("\n"+"="*70)
print("Interpretation:")
print("  1. PART 0 If delta_all approx delta_mal -> Filtering ineffective, fix before proceeding.")
print("  2. If MES-like has highest EV and tumour-level EV-Hypoxia correlation is significant -> Hypoxia chain closed.")
print("  3. If all states show tmc > para -> Not driven by state composition, but a broad regional effect.")
print("  4. If significant only in MES-like -> Gradient is driven by hypoxic state, mechanism clear.")