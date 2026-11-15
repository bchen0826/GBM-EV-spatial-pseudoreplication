#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Pseudoreplication Boundaries in Spatial Transcriptomic Inference: A Formal Benchmark (v2)
=============================================================================
Upgrades the descriptive warning "effective sample size inflated by ~1980x" 
into a generalizable methodological benchmark.

Core mechanistic distinction (validated in v1 simulation):
  PAN vs LE is an [within-tumour contrast] -- PAN and LE are within the same tumour,
  the tumour random intercept u_t is added to both, cancelling out when taking the difference.
  Therefore, ICC is **not** the primary threat; **spatial autocorrelation** is.
  [Between-tumour contrasts] (e.g., tumour-level covariates vs expression) are driven by ICC.
  
  This is one of the most important methodological conclusions of this paper, 
  and it defines the boundary of "when spot-level inference fails".

Output:
  1. Nominal inflation upper bound (observations / independent units)
  2. Within-tumour Type I error driven by spatial autocorrelation
  3. Between-tumour Type I error driven by ICC
  4. Power comparison incorporating between-tumour effect heterogeneity, answering "is aggregation too conservative?"
  5. Real data: three methods side-by-side + permutation control

Usage:
  python3 pseudoreplication_benchmark_v2.0.py
  python3 pseudoreplication_benchmark_v2.0.py --real spot_level_scores.csv
=============================================================================
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu, wilcoxon, binomtest, pearsonr

warnings.filterwarnings("ignore")
ALPHA = 0.05

# =============================================================================
# 1. Nominal inflation upper bound
# =============================================================================
def nominal_inflation(n_spots, n_units):
    """Observations / independent units. This is the upper bound, not the actual design effect."""
    return n_spots / n_units

def design_effect(m, icc):
    """Kish design effect. Only applicable to [between-tumour contrasts]."""
    return 1.0 + (m - 1.0) * icc

# =============================================================================
# 2. Spatial autocorrelation field
# =============================================================================
def spatial_field(n, rho, rng):
    """rho=0 independent noise; higher rho means adjacent spots are more similar."""
    side = int(np.ceil(np.sqrt(n)))
    g = rng.normal(0, 1, (side, side))
    if rho > 0:
        g = ndimage.gaussian_filter(g, sigma=rho * 4.0)
    g = g / (g.std() + 1e-9)
    return g.ravel()[:n]

# =============================================================================
# 3. Data simulation
# =============================================================================
def simulate_within(n_tumour, n_spot, icc, rho, delta, tau_delta, rng, pan_frac=0.30):
    """
    [Within-tumour contrast] PAN vs LE in the same tumour.
    tau_delta = between-tumour heterogeneity of the true gradient (sd). This is why we observed 8/9 instead of 9/9.
    """
    n_pan = max(5, int(n_spot * pan_frac))
    n_le = max(5, n_spot - n_pan)
    u = rng.normal(0, np.sqrt(icc), n_tumour)
    delta_t = delta + rng.normal(0, tau_delta, n_tumour)   # True gradient per tumour
    within_sd = np.sqrt(max(1e-9, 1.0 - icc))

    recs = []
    for t in range(n_tumour):
        comp = np.r_[np.ones(n_pan), np.zeros(n_le)]
        eps = spatial_field(n_pan + n_le, rho, rng) * within_sd
        y = delta_t[t] * comp + u[t] + eps
        recs.append(pd.DataFrame({
            "tumour": f"T{t}",
            "compartment": np.where(comp == 1, "PAN", "LE"),
            "y": y}))
    return pd.concat(recs, ignore_index=True)

def simulate_between(n_tumour, n_spot, icc, rho, beta, rng):
    """
    [Between-tumour contrast] Relationship between tumour-level covariate X and expression y. Driven by ICC.
    """
    X = rng.normal(0, 1, n_tumour)
    u = rng.normal(0, np.sqrt(icc), n_tumour)
    within_sd = np.sqrt(max(1e-9, 1.0 - icc))
    recs = []
    for t in range(n_tumour):
        eps = spatial_field(n_spot, rho, rng) * within_sd
        y = beta * X[t] + u[t] + eps
        recs.append(pd.DataFrame({"tumour": f"T{t}", "X": X[t], "y": y}))
    return pd.concat(recs, ignore_index=True)

# =============================================================================
# 4. Test methods
# =============================================================================
def test_spot_within(df):
    a = df.loc[df.compartment == "PAN", "y"].values
    b = df.loc[df.compartment == "LE", "y"].values
    return ttest_ind(a, b, equal_var=False).pvalue

def test_agg_within(df, method="sign"):
    m = df.groupby(["tumour", "compartment"])["y"].mean().unstack()[["PAN", "LE"]].dropna()
    d = (m["PAN"] - m["LE"]).values
    if len(d) < 3:
        return np.nan
    if method == "t":
        return ttest_1samp(d, 0).pvalue
    if method == "wilcoxon":
        return wilcoxon(d).pvalue
    return binomtest(int((d > 0).sum()), len(d), 0.5).pvalue

def test_spot_between(df):
    return pearsonr(df["X"].values, df["y"].values).pvalue

def test_agg_between(df):
    g = df.groupby("tumour").agg(X=("X", "mean"), y=("y", "mean"))
    if len(g) < 3:
        return np.nan
    return pearsonr(g["X"].values, g["y"].values).pvalue

def test_mixedlm_within(df):
    import statsmodels.formula.api as smf
    d = df.copy()
    d["pan"] = (d.compartment == "PAN").astype(float)
    try:
        fit = smf.mixedlm("y ~ pan", d, groups=d["tumour"]).fit(reml=True, method="lbfgs")
        return float(fit.pvalues["pan"]), float(fit.params["pan"]), float(fit.bse["pan"])
    except Exception:
        return np.nan, np.nan, np.nan

# =============================================================================
# 5. Type I error: Within-tumour (driven by spatial autocorrelation)
# =============================================================================
def t1e_within(scenarios, B, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for sc in scenarios:
        n_t, n_s, icc, rho = sc["n_tumour"], sc["n_spot"], sc["icc"], sc.get("rho", 0.0)
        hit = {"spot": 0, "agg_t": 0, "agg_sign": 0, "agg_w": 0}
        ok = 0
        for _ in range(B):
            df = simulate_within(n_t, n_s, icc, rho, delta=0.0, tau_delta=0.0, rng=rng)
            p1 = test_spot_within(df)
            p2 = test_agg_within(df, "t")
            p3 = test_agg_within(df, "sign")
            p4 = test_agg_within(df, "wilcoxon")
            if np.isnan(p1) or np.isnan(p2) or np.isnan(p3):
                continue
            ok += 1
            hit["spot"] += p1 < ALPHA; hit["agg_t"] += p2 < ALPHA
            hit["agg_sign"] += p3 < ALPHA; hit["agg_w"] += p4 < ALPHA
        rows.append({"contrast": "within", "n_tumour": n_t, "spots_per_tumour": n_s,
                     "ICC": icc, "spatial_rho": rho,
                     "T1E_spot": hit["spot"] / ok, "T1E_agg_t": hit["agg_t"] / ok,
                     "T1E_agg_sign": hit["agg_sign"] / ok, "T1E_agg_wilcoxon": hit["agg_w"] / ok,
                     "B": ok})
        print(f"  within  rho={rho} ICC={icc} n={n_t} -> spot={rows[-1]['T1E_spot']:.3f} "
              f"agg_sign={rows[-1]['T1E_agg_sign']:.3f}")
    return pd.DataFrame(rows)

# =============================================================================
# 6. Type I error: Between-tumour (driven by ICC)
# =============================================================================
def t1e_between(scenarios, B, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for sc in scenarios:
        n_t, n_s, icc, rho = sc["n_tumour"], sc["n_spot"], sc["icc"], sc.get("rho", 0.0)
        hit = {"spot": 0, "agg": 0}
        ok = 0
        for _ in range(B):
            df = simulate_between(n_t, n_s, icc, rho, beta=0.0, rng=rng)
            p1 = test_spot_between(df); p2 = test_agg_between(df)
            if np.isnan(p1) or np.isnan(p2):
                continue
            ok += 1
            hit["spot"] += p1 < ALPHA; hit["agg"] += p2 < ALPHA
        rows.append({"contrast": "between", "n_tumour": n_t, "spots_per_tumour": n_s,
                     "ICC": icc, "spatial_rho": rho,
                     "design_effect": design_effect(n_s, icc),
                     "T1E_spot": hit["spot"] / ok, "T1E_aggregate": hit["agg"] / ok, "B": ok})
        print(f"  between ICC={icc} n={n_t} spots={n_s} -> spot={rows[-1]['T1E_spot']:.3f} "
              f"agg={rows[-1]['T1E_aggregate']:.3f}")
    return pd.DataFrame(rows)

# =============================================================================
# 7. Power (incorporating between-tumour effect heterogeneity)
# =============================================================================
def power_within(deltas, n_tumour, n_spot, icc, rho, tau_delta, B, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for delta in deltas:
        hit = {"spot": 0, "agg_t": 0, "agg_sign": 0}
        ok = 0
        for _ in range(B):
            df = simulate_within(n_tumour, n_spot, icc, rho, delta, tau_delta, rng)
            p1 = test_spot_within(df); p2 = test_agg_within(df, "t"); p3 = test_agg_within(df, "sign")
            if np.isnan(p1) or np.isnan(p2) or np.isnan(p3):
                continue
            ok += 1
            hit["spot"] += p1 < ALPHA; hit["agg_t"] += p2 < ALPHA; hit["agg_sign"] += p3 < ALPHA
        rows.append({"true_delta": delta, "tau_delta": tau_delta,
                     "power_spot": hit["spot"] / ok, "power_agg_t": hit["agg_t"] / ok,
                     "power_agg_sign": hit["agg_sign"] / ok, "B": ok})
        print(f"  Delta={delta:.2f} tau={tau_delta:.2f} -> spot={rows[-1]['power_spot']:.2f} "
              f"agg_t={rows[-1]['power_agg_t']:.2f} sign={rows[-1]['power_agg_sign']:.2f}")
    return pd.DataFrame(rows)

# =============================================================================
# 8. Real data
# =============================================================================
def real_data(path, ev="EV_score", tumour="sample", comp="compartment", n_perm=1000, seed=11):
    df = pd.read_csv(path)[[tumour, comp, ev]].rename(
        columns={tumour: "tumour", comp: "compartment", ev: "y"}).dropna()
    out = {}
    n_spot, n_tumour = len(df), df["tumour"].nunique()
    print(f"\nReal data: {n_spot} spots, {n_tumour} tumours")

    a = df.loc[df.compartment == "PAN", "y"].values
    b = df.loc[df.compartment == "LE", "y"].values
    out["A_spot_level"] = dict(method="Mann-Whitney (spot-level)", n_PAN=len(a), n_LE=len(b),
                               p=float(mannwhitneyu(a, b).pvalue),
                               median_diff=float(np.median(a) - np.median(b)))

    m = df.groupby(["tumour", "compartment"])["y"].mean().unstack()[["PAN", "LE"]].dropna()
    d = (m["PAN"] - m["LE"]).values
    k, n = int((d > 0).sum()), len(d)
    out["B_aggregate"] = dict(n_tumours=n, concordant=k,
                              p_sign=float(binomtest(k, n, 0.5).pvalue),
                              p_wilcoxon=float(wilcoxon(d).pvalue),
                              p_t=float(ttest_1samp(d, 0).pvalue),
                              median_delta=float(np.median(d)),
                              IQR=[float(np.percentile(d, 25)), float(np.percentile(d, 75))])

    p_mix, beta, se = test_mixedlm_within(df)
    out["C_mixed_model"] = dict(method="LMM y ~ PAN + (1|tumour)",
                                beta=beta, se=se, p=p_mix)

    # Nominal inflation + ICC estimation (for reference only, design effect applies to between-tumour contrasts)
    m_per = n_spot / n_tumour
    grand = df["y"].mean()
    ssb = df.groupby("tumour")["y"].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    sst = ((df["y"] - grand) ** 2).sum()
    icc_hat = float(np.clip(ssb / sst, 1e-4, 0.99))
    out["D_inflation"] = dict(n_spots=n_spot, n_tumours=n_tumour,
                              nominal_inflation=float(nominal_inflation(n_spot, n_tumour)),
                              ICC_hat=icc_hat,
                              design_effect_between=float(design_effect(m_per, icc_hat)),
                              note="Within-tumour contrasts are dominated by spatial autocorrelation; design effect only applies to between-tumour contrasts")

    rng = np.random.default_rng(seed)
    obs = np.median(a) - np.median(b)
    labels = df["compartment"].values.copy()
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        aa = df["y"].values[perm == "PAN"]; bb = df["y"].values[perm == "LE"]
        if len(aa) > 5 and len(bb) > 5:
            null.append(np.median(aa) - np.median(bb))
    null = np.array(null)
    out["E_permutation"] = dict(observed=float(obs), null_median=float(np.median(null)),
                                null_p95=float(np.percentile(null, 95)),
                                empirical_p=float((np.abs(null) >= abs(obs)).mean()))
    return out

# =============================================================================
# 9. Main workflow
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    B = 150 if a.quick else 400
    PB = 100 if a.quick else 250

    print("=" * 74)
    print("  Pseudoreplication Formal Benchmark v2 -- Differentiating Within- and Between-Tumour Contrasts")
    print("=" * 74)

    print("\n[0] Nominal inflation upper bound (observations / independent units)")
    for n_spot, n_t, tag in [(23762, 12, "GSE237183 as 12 units"),
                             (23762, 5, "GSE237183 as 5 tumours"),
                             (15000, 9, "Final 9 cases in this study")]:
        print(f"  {tag}: {n_spot:,} / {n_t} = {nominal_inflation(n_spot, n_t):,.0f}x")
    print("  Note: This is an upper bound, not the actual design effect.")

    print(f"\n[1] Within-tumour contrast (PAN vs LE) Type I error  B={B}")
    print("  Mechanism: Tumour random intercept cancels out -> ICC is not the threat; spatial autocorrelation is.")
    sc_w = [dict(n_tumour=16, n_spot=1000, icc=0.20, rho=0.0),
            dict(n_tumour=16, n_spot=1000, icc=0.20, rho=0.25),
            dict(n_tumour=16, n_spot=1000, icc=0.20, rho=0.5),
            dict(n_tumour=16, n_spot=1000, icc=0.20, rho=0.8),
            dict(n_tumour=16, n_spot=3000, icc=0.20, rho=0.5),
            dict(n_tumour=9, n_spot=1000, icc=0.20, rho=0.5)]
    w = t1e_within(sc_w, B)
    w.to_csv("t1e_within.csv", index=False)
    print("\n" + w.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n[2] Between-tumour contrast (tumour-level covariate) Type I error  B={B}")
    print("  Mechanism: Driven by ICC, spot-level tests are severely inflated.")
    sc_b = [dict(n_tumour=16, n_spot=1000, icc=0.05, rho=0.0),
            dict(n_tumour=16, n_spot=1000, icc=0.20, rho=0.0),
            dict(n_tumour=16, n_spot=1000, icc=0.50, rho=0.0),
            dict(n_tumour=16, n_spot=3000, icc=0.20, rho=0.0)]
    bdf = t1e_between(sc_b, B)
    bdf.to_csv("t1e_between.csv", index=False)
    print("\n" + bdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n[3] Validation: Under balanced design, aggregate t-test approx mixed model")
    rng = np.random.default_rng(99)
    dv = simulate_within(16, 300, 0.20, 0.3, delta=0.31, tau_delta=0.15, rng=rng)
    p_agg = test_agg_within(dv, "t")
    p_mix, beta, se = test_mixedlm_within(dv)
    print(f"  Aggregate t-test p = {p_agg:.6f}")
    print(f"  Mixed model      p = {p_mix:.6f}  (beta={beta:+.4f}, se={se:.4f})")

    print(f"\n[4] Power: Impact of between-tumour effect heterogeneity tau_delta  B={PB}")
    print("  n=9 (actual sample size in this study), Delta=0.31")
    for tau in [0.0, 0.15, 0.30, 0.50]:
        pw = power_within([0.31], 9, 1000, 0.20, 0.3, tau, PB)
        r = pw.iloc[0]
        print(f"    tau={tau:.2f}: Sign test power={r['power_agg_sign']:.2f}  "
              f"Aggregate t={r['power_agg_t']:.2f}  Spot-level={r['power_spot']:.2f}")

    print("\n  n=16 control:")
    for tau in [0.0, 0.30]:
        pw = power_within([0.31], 16, 1000, 0.20, 0.3, tau, PB)
        r = pw.iloc[0]
        print(f"    tau={tau:.2f}: Sign test power={r['power_agg_sign']:.2f}")

    pw_full = power_within([0.10, 0.20, 0.31, 0.50], 9, 1000, 0.20, 0.3, 0.30, PB)
    pw_full.to_csv("power_within_n9.csv", index=False)
    print("\n" + pw_full.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if a.real:
        print(f"\n[5] Real data: {a.real}")
        for k, v in real_data(a.real).items():
            print(f"\n  --- {k} ---")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), dpi=300)

        ax[0].plot(w["spatial_rho"], w["T1E_spot"], "o-", color="#C0392B", label="Spot-level")
        ax[0].plot(w["spatial_rho"], w["T1E_agg_sign"], "s-", color="#2471A3", label="Tumour-level (sign)")
        ax[0].axhline(0.05, color="k", ls="--", lw=1.2)
        ax[0].set_xlabel("Spatial autocorrelation (rho)")
        ax[0].set_ylabel("Type I error"); ax[0].set_title("Within-tumour (PAN vs LE)", fontsize=11)
        ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)

        ax[1].bar(np.arange(len(bdf)) - 0.2, bdf["T1E_spot"], 0.4, color="#C0392B", label="Spot-level")
        ax[1].bar(np.arange(len(bdf)) + 0.2, bdf["T1E_aggregate"], 0.4, color="#2471A3", label="Tumour-level")
        ax[1].axhline(0.05, color="k", ls="--", lw=1.2)
        ax[1].set_xticks(np.arange(len(bdf)))
        ax[1].set_xticklabels([f"ICC={r.ICC}\n{int(r.spots_per_tumour)}sp" for r in bdf.itertuples()], fontsize=8)
        ax[1].set_ylabel("Type I error"); ax[1].set_title("Between-tumour", fontsize=11)
        ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)

        ax[2].plot(pw_full["true_delta"], pw_full["power_spot"], "o-", color="#C0392B", label="Spot-level")
        ax[2].plot(pw_full["true_delta"], pw_full["power_agg_t"], "s-", color="#2471A3", label="Aggregate (t)")
        ax[2].plot(pw_full["true_delta"], pw_full["power_agg_sign"], "^--", color="#7D3C98", label="Aggregate (sign)")
        ax[2].axhline(0.8, color="gray", ls=":", lw=1)
        ax[2].set_xlabel("True Delta"); ax[2].set_ylabel("Power")
        ax[2].set_title("Power (n=9, tau=0.30)", fontsize=11)
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

        for x in ax: x.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("pseudoreplication_benchmark.png", dpi=300)
        plt.savefig("pseudoreplication_benchmark.pdf")
        print("\nFigure saved: pseudoreplication_benchmark.png / .pdf")
    except Exception as e:
        print(f"\nPlotting failed: {e}")

    print("\n" + "=" * 74)
    print("Outputs: t1e_within.csv / t1e_between.csv / power_within_n9.csv / Figures")
    print("=" * 74)

if __name__ == "__main__":
    main()