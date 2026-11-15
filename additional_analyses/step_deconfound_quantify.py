# -*- coding: utf-8 -*-
"""
Quantitative report of deconfounding + Final piece of the puzzle (MES-like / Hypoxia state)

All numbers output here are ready to be written into the manuscript: effect size + CI, attenuation test, specificity control.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr

rng = np.random.default_rng(0)
df = pd.read_csv("gbm_full_metadata.csv")
CT  = "cell_type"      # Modify according to PART 0 output
MAL = "Malignant"      # Modify according to PART 0 output

def pair_delta(sub, val="EV_score"):
    m = sub.groupby(["patient","region"])[val].mean().unstack()
    m = m[["tmc","para"]].dropna()
    return m["tmc"] - m["para"]

def boot_ci(d, B=10000):
    bs = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(B)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def report(d, label):
    k, n = int((d>0).sum()), len(d)
    lo, hi = boot_ci(d)
    print(f"\n{label}")
    print(f"  n = {n} paired patients | Concordant {k}/{n} ({k/n:.1%})")
    print(f"  Median Delta = {np.median(d):+.4f}   95%CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  Sign test p = {binomtest(k,n,0.5).pvalue:.5f}")
    print(f"  Wilcoxon  p = {wilcoxon(d).pvalue:.5f}")
    return d

# ---------- 1. Two-level effect sizes (Must report, not just p-values) ----------
print("="*72); print("1  Two-level Effect Sizes and Confidence Intervals"); print("="*72)
d_all = report(pair_delta(df), "[All cells] tmc - para")
mal   = df[df[CT].eq(MAL)]
d_mal = report(pair_delta(mal), "[Malignant cells] tmc - para")

# ---------- 2. Attenuation test (Correct approach, replacing 'one significant, one not') ----------
print("\n"+"="*72); print("2  Attenuation test: Delta_all - Delta_mal per patient"); print("="*72)
common = d_all.index.intersection(d_mal.index)
decay  = d_all[common] - d_mal[common]
k, n = int((decay>0).sum()), len(decay)
lo, hi = boot_ci(decay.values)
print(f"  n = {n} | Positive attenuation in {k}/{n}")
print(f"  Median attenuation = {np.median(decay):+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  Sign test p = {binomtest(k,n,0.5).pvalue:.5f}")
print(f"  Wilcoxon  p = {wilcoxon(decay).pvalue:.5f}")
print(f"  Attenuation proportion = 1 - median(Delta_mal)/median(Delta_all) = {1-np.median(d_mal)/np.median(d_all):.1%}")
print("\n  Interpretation: This step is the evidence for 'confounding driven'.")
print("                  A p(Delta_mal)=0.27 alone is only 'not detected', cannot infer 'driven by composition'.")

# ---------- 3. Tactical B: Malignant proportion vs EV (with specificity control) ----------
print("\n"+"="*72); print("3  Malignant cell proportion vs EV score (Tactical B test)"); print("="*72)
prop = df.groupby(["patient","region"]).apply(
    lambda s: pd.Series({"mal_prop": s[CT].eq(MAL).mean(),
                         "EV": s["EV_score"].mean()})).reset_index()
r = spearmanr(prop["mal_prop"], prop["EV"])
print(f"  Spearman rho = {r.statistic:.3f}, p = {r.pvalue:.3e}, n = {len(prop)}")
print("  [WARN] Must perform random gene set control: if random set is also significant, it's just a 'tumor content' effect, not EV-specific.")

# ---------- 4. Final piece: State axis within malignant cells ----------
print("\n"+"="*72); print("4  Within malignant cells: MES-like / Hypoxia state axis"); print("="*72)
HYPOXIA = ["CA9","VEGFA","SLC2A1","PGK1","LDHA","ALDOA","ENO1","ADM","NDRG1","P4HA1",
           "PLOD2","BNIP3","DDIT4","ANKRD37","PDK1","HK2","PFKFB3","PFKFB4","EGLN3","HILPDA"]
MES = ["CD44","CHI3L1","CHI3L2","MET","TIMP1","FN1","COL1A1","SERPINE1","ANXA2","VIM",
       "LGALS1","S100A10","EMP3","TGFBI","TAGLN","ACTA2","CRYAB","HSPA1A","HSPA1B","DNAJB1",
       "HSPB1","JUN","JUNB","FOS","EGR1","ATF3","SOD2","TXN","GPX1","HMOX1","SLC2A1","PGK1"]

def add_score(sub, genes, name):
    g = [x for x in genes if x in sub.columns]
    if len(g) < 5:
        print(f"  [FAIL] {name}: Only {len(g)} genes available in matrix -> Requires raw expression matrix, not metadata"); return None
    z = sub[g].apply(lambda c: (c - c.mean())/(c.std()+1e-9))
    sub[name] = z.mean(axis=1)
    print(f"  [OK] {name}: {len(g)} genes"); return name

for nm, gs in [("Hypoxia", HYPOXIA), ("MES", MES)]:
    if add_score(mal, gs, nm+"_score"):
        # Tumour level: EV vs state score (anti-pseudoreplication, one value per tumour)
        pt = mal.groupby("patient").apply(
            lambda s: pd.Series({"ev": s["EV_score"].mean(), "st": s[nm+"_score"].mean()}))
        rr = spearmanr(pt["ev"], pt["st"])
        print(f"     Tumour-level EV vs {nm}: rho={rr.statistic:.3f}, p={rr.pvalue:.4f}, n={len(pt)}")
        # Intra-tumour quartile: EV difference between highest/lowest quartile cells in each tumour
        qs=[]
        for pid, s in mal.groupby("patient"):
            hi_q = s[s[nm+"_score"] >= s[nm+"_score"].quantile(0.75)]["EV_score"].mean()
            lo_q = s[s[nm+"_score"] <= s[nm+"_score"].quantile(0.25)]["EV_score"].mean()
            qs.append(hi_q - lo_q)
        qs=np.array(qs); kk=int((qs>0).sum())
        print(f"     Quartile contrast High-{nm} minus Low-{nm} EV diff: {kk}/{len(qs)} concordant, "
              f"sign test p={binomtest(kk,len(qs),0.5).pvalue:.5f}, median={np.median(qs):+.4f}")

print("\n"+"="*72)
print("Result Interpretation:")
print("  - Step 2 significant + Step 1 Delta_mal CI crosses 0 -> Can write 'confounding driven'")
print("  - Step 4 MES/Hypoxia quartile significant -> Hypoxia chain established at the state level (final chance)")
print("  - Step 4 not significant -> Hypoxia chain not closed, manuscript should follow negative/methodological route")