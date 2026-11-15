#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type I error benchmark expressed directly in residual Moran's I.
================================================================

Companion script to pseudoreplication_benchmark.py.

Why this script exists
----------------------
The main benchmark parameterises spatial autocorrelation with a simulation
parameter rho (Gaussian smoothing width). rho is not directly measurable in
real data, so a reviewer cannot locate a real dataset on that axis. This
script re-expresses the same benchmark on the axis of residual Moran's I,
which can be measured on any Visium section with spot coordinates.

For each target level of spatial autocorrelation it simulates hierarchical
spatial transcriptomic data under the NULL hypothesis (true PAN-LE effect
= 0), and records the empirical type I error rate of three inference
strategies:

  (i)   spot-level Welch t test  (all spots treated as independent)
  (ii)  random-intercept linear mixed model, y ~ PAN + (1 | tumour)
  (iii) tumour-level aggregation followed by an exact sign test

Nominal alpha = 0.05. Output: t1e_by_moran.csv

Usage
-----
  python moran_type1_benchmark.py            # full run (B=100 per level)
  python moran_type1_benchmark.py --quick   # smoke test  (B=25 per level)

To locate your own data on this curve, compute residual Moran's I per
tumour (observed score minus the mean of that tumour x compartment stratum),
take the median across tumours, and interpolate in t1e_by_moran.csv.
Residuals must be computed within tumour x compartment: concatenating spots
across tumours creates artificial spatial structure.
"""
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import ttest_ind, ttest_1samp
from scipy.spatial import cKDTree
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
ALPHA = 0.05

# Empirically calibrated smoothing widths (sigma of the Gaussian filter)
# that produce the target residual Moran's I on a 40x40 square lattice
# with k=6 nearest neighbours (Visium hexagonal geometry).
SIGMA_LUT = {0.0: 0.20, 0.2: 0.60, 0.4: 0.75, 0.6: 0.85, 0.8: 1.20}


def spatial_field(n, sigma, rng):
    """Spatially autocorrelated standardised noise field."""
    side = int(np.ceil(np.sqrt(n)))
    g = rng.normal(0, 1, (side + 4, side + 4))
    if sigma > 0:
        g = ndimage.gaussian_filter(g, sigma=sigma)
    g = g[:side, :side]
    g = g / (g.std() + 1e-9)
    return g.ravel()[:n]


def morans_i(values, coords, k=6):
    """Moran's I under k-nearest-neighbour row-standardised weights."""
    v = np.asarray(values, float)
    xy = np.asarray(coords, float)
    n = len(v)
    if n < 10:
        return np.nan
    z = v - v.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return np.nan
    _, idx = cKDTree(xy).query(xy, k=min(k + 1, n))
    num, W = 0.0, 0.0
    for i in range(n):
        nb = np.atleast_1d(idx[i])
        nb = nb[nb != i]
        if len(nb) == 0:
            continue
        num += z[i] * z[nb].sum()
        W += len(nb)
    return float((n / W) * (num / denom)) if W > 0 else np.nan


def simulate(n_tumour, n_spot, icc, sigma, delta, tau_delta, rng, pan_frac=0.30):
    """Hierarchical data: tumour -> spot, with a within-tumour PAN/LE contrast."""
    n_pan = max(5, int(n_spot * pan_frac))
    n_le = max(5, n_spot - n_pan)
    u = rng.normal(0, np.sqrt(icc), n_tumour)
    delta_t = delta + rng.normal(0, tau_delta, n_tumour)
    within_sd = np.sqrt(max(1e-9, 1.0 - icc))
    recs = []
    for t in range(n_tumour):
        comp = np.r_[np.ones(n_pan), np.zeros(n_le)]
        eps = spatial_field(n_pan + n_le, sigma, rng) * within_sd
        y = delta_t[t] * comp + u[t] + eps
        recs.append(pd.DataFrame({
            "tumour": f"T{t}",
            "compartment": np.where(comp == 1, "PAN", "LE"),
            "y": y,
        }))
    return pd.concat(recs, ignore_index=True)


def p_spot(df):
    a = df.loc[df.compartment == "PAN", "y"].values
    b = df.loc[df.compartment == "LE", "y"].values
    return ttest_ind(a, b, equal_var=False).pvalue


def p_aggregate_sign(df):
    m = df.groupby(["tumour", "compartment"])["y"].mean().unstack()[["PAN", "LE"]].dropna()
    d = (m["PAN"] - m["LE"]).values
    if len(d) < 3:
        return np.nan
    return binomtest(int((d > 0).sum()), len(d), 0.5).pvalue


def p_mixedlm(df):
    import statsmodels.formula.api as smf
    d = df.copy()
    d["pan"] = (d.compartment == "PAN").astype(float)
    try:
        fit = smf.mixedlm("y ~ pan", d, groups=d["tumour"]).fit(reml=True, method="lbfgs")
        return float(fit.pvalues["pan"])
    except Exception:
        return np.nan


def measured_moran(sigma, n=1600, reps=8, seed=11):
    """Measure the Moran's I actually produced by a given smoothing width."""
    rng = np.random.default_rng(seed)
    side = int(np.sqrt(n))
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.c_[gx.ravel(), gy.ravel()]
    return float(np.nanmean([morans_i(spatial_field(n, sigma, rng), coords) for _ in range(reps)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    B = 25 if a.quick else 100

    print("Type I error benchmark on the residual Moran's I axis")
    print(f"n_tumour=16, spots_per_tumour=800, ICC=0.20, B={B}, alpha={ALPHA}\n")

    rng = np.random.default_rng(2026)
    rows = []
    for target, sigma in SIGMA_LUT.items():
        mi = measured_moran(sigma)
        hit = {"spot": 0, "mixed": 0, "agg": 0}
        ok = 0
        for _ in range(B):
            df = simulate(16, 800, 0.20, sigma, delta=0.0, tau_delta=0.0, rng=rng)
            p1, p2, p3 = p_spot(df), p_aggregate_sign(df), p_mixedlm(df)
            if np.isnan(p1) or np.isnan(p2) or np.isnan(p3):
                continue
            ok += 1
            hit["spot"] += p1 < ALPHA
            hit["mixed"] += p3 < ALPHA
            hit["agg"] += p2 < ALPHA
        rows.append({
            "target_moran": target,
            "measured_moran": round(mi, 3),
            "sigma": sigma,
            "T1E_spot": round(hit["spot"] / ok, 3),
            "T1E_mixed": round(hit["mixed"] / ok, 3),
            "T1E_agg_sign": round(hit["agg"] / ok, 3),
            "B": ok,
        })
        r = rows[-1]
        print(f"  Moran={r['measured_moran']:.3f}  spot={r['T1E_spot']:.3f}  "
              f"mixed={r['T1E_mixed']:.3f}  agg={r['T1E_agg_sign']:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv("t1e_by_moran.csv", index=False)
    print("\nSaved t1e_by_moran.csv")


if __name__ == "__main__":
    main()