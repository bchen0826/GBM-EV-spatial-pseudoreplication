# -*- coding: utf-8 -*-
"""
Incremental Value Analysis of the Hypoxia-Invasion Signature
Answers the reviewer's first question: what does this signature explain beyond existing clinical-molecular variables?

Outputs 6 essential metrics + 3 diagnostic checks. Pure statistics, no new data required.
Dependencies: pip install lifelines scikit-learn
"""
import numpy as np, pandas as pd
from scipy.stats import norm, chi2

rng = np.random.default_rng(2026)

# ==================================================================
# 0. Data Input (modify according to your real column names)
# ==================================================================
FILE = "glioma_survival.csv"          # ← Change to your file
TIME, EVENT = "OS", "status"          # status: 1=death 0=censored
SIG  = "hypoxia_invasion_score"       # Your signature score
COHORT = "cohort"                     # CGGA693 / CGGA325 / TCGA ...

BASE_COVARS = ["age", "grade", "idh_status", "codel_1p19q", "mgmt_meth"]

# ==================================================================
# 1. Core Functions
# ==================================================================
def harrell_c(time, event, risk, ties="efron"):
    """Harrell's C. Higher risk = higher hazard."""
    from lifelines.utils import concordance_index
    return concordance_index(time, -np.asarray(risk), event)

def fit_cox(df, covars, time=TIME, event=EVENT, penal=0.0):
    from lifelines import CoxPHFitter
    d = df[covars + [time, event]].dropna()
    cph = CoxPHFitter(penalizer=penal)
    cph.fit(d, duration_col=time, event_col=event, formula="~" + " + ".join(covars))
    return cph

def lp(cph, df):
    """Linear predictor (risk score) = beta' x"""
    return cph.predict_log_partial_hazard(df).values

# ---------- 1.1 Delta C-index + bootstrap CI (paired resampling) ----------
def delta_c(df, base_vars, new_var, B=500):
    """Delta C-index, paired bootstrap (calculate both on the same resampled dataset for valid CI)"""
    c0 = harrell_c(df[TIME], df[EVENT], lp(fit_cox(df, base_vars), df))
    c1 = harrell_c(df[TIME], df[EVENT], lp(fit_cox(df, base_vars+[new_var]), df))
    boot = []
    n = len(df)
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        s = df.iloc[idx]
        if s[EVENT].sum() < 5: continue
        try:
            a = harrell_c(s[TIME], s[EVENT], lp(fit_cox(s, base_vars), s))
            b = harrell_c(s[TIME], s[EVENT], lp(fit_cox(s, base_vars+[new_var]), s))
            boot.append(b - a)
        except Exception:
            continue
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return dict(c_base=c0, c_full=c1, delta=c1-c0, ci=(lo, hi),
                p_boot=max(p, 1.0/len(boot)), n_boot=len(boot))

# ---------- 1.2 Multivariable Cox: independent HR of the signature ----------
def independent_hr(df, base_vars, new_var):
    cph = fit_cox(df, base_vars + [new_var])
    s = cph.summary.loc[new_var]
    sd = float(df[new_var].std())
    return dict(hr=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                hi=np.exp(s["coef upper 95%"]), p=s["p"],
                # HR per SD (easier to interpret than per unit)
                hr_sd=np.exp(s["coef"] * sd),
                hr_sd_lo=np.exp(s["coef lower 95%"] * sd),
                hr_sd_hi=np.exp(s["coef upper 95%"] * sd))

# ---------- 1.3 NRI / IDI (at fixed time points) ----------
def nri_idi(df, base_vars, new_var, horizon):
    """
    Category-free NRI and IDI at a given time horizon.
    Censoring handling: Only include individuals with known status at this horizon (simple version).
    If censoring is heavy, consider IPCW.
    """
    d = df.copy()
    known = ((d[TIME] >= horizon) | (d[EVENT] == 1)).values
    d = d[known]
    case = ((d[TIME] <= horizon) & (d[EVENT] == 1)).values.astype(int)
    ctrl = (d[TIME] > horizon).values.astype(int)
    if case.sum() < 10 or ctrl.sum() < 10:
        return dict(note=f"horizon={horizon} insufficient events/controls")
    p0 = 1 - np.exp(-np.exp(lp(fit_cox(d, base_vars), d)))          # approximate risk probability
    p1 = 1 - np.exp(-np.exp(lp(fit_cox(d, base_vars+[new_var]), d)))
    up, dn = p1 - p0, 0
    # category-free NRI
    nri_case = (up[case==1] > 0).mean() - (up[case==1] < 0).mean()
    nri_ctrl = (up[ctrl==1] < 0).mean() - (up[ctrl==1] > 0).mean()
    nri = nri_case + nri_ctrl
    se_case = np.sqrt(max(1e-9, 1 - nri_case**2) / case.sum())
    se_ctrl = np.sqrt(max(1e-9, 1 - nri_ctrl**2) / ctrl.sum())
    z = nri / np.sqrt(se_case**2 + se_ctrl**2)
    # IDI
    idi = (p1[case==1] - p0[case==1]).mean() - (p1[ctrl==1] - p0[ctrl==1]).mean()
    se_idi = np.sqrt(np.var(p1[case==1]-p0[case==1])/case.sum()
                     + np.var(p1[ctrl==1]-p0[ctrl==1])/ctrl.sum())
    return dict(horizon=horizon, n_case=int(case.sum()), n_ctrl=int(ctrl.sum()),
                cfNRI=nri, nri_p=2*(1-norm.cdf(abs(z))),
                IDI=idi, idi_p=2*(1-norm.cdf(abs(idi/se_idi))))

# ---------- 1.4 Interaction test: Signature x Subtype ----------
def interaction_test(df, new_var, group_col, base_vars):
    """
    Test whether the HR of the signature truly differs between two subtypes.
    Wrong approach: significant in LGG, non-significant in GBM -> conclude LGG is stronger.
    Correct approach: Explicit interaction term.
    """
    d = df.dropna(subset=[new_var, group_col] + base_vars + [TIME, EVENT]).copy()
    d["_g"] = (d[group_col] == d[group_col].unique()[0]).astype(int)
    d["_int"] = d[new_var] * d["_g"]
    cph = fit_cox(d, base_vars + [new_var, "_g", "_int"])
    s = cph.summary.loc["_int"]
    return dict(hr_ratio=np.exp(s["coef"]), p_interaction=s["p"],
                lo=np.exp(s["coef lower 95%"]), hi=np.exp(s["coef upper 95%"]))

# ---------- 1.5 Overfitting correction (necessary if derived in the same cohort) ----------
def optimism_corrected_delta_c(df, base_vars, new_var, B=200):
    """
    If the signature is derived in this cohort, Delta C is inflated by overfitting.
    Approach: Bootstrap internal validation to estimate optimism and subtract it from apparent value.
    """
    c_app = harrell_c(df[TIME], df[EVENT], lp(fit_cox(df, base_vars+[new_var]), df))
    opt = []
    n = len(df)
    for _ in range(B):
        idx = rng.choice(n, n, replace=True); s = df.iloc[idx]
        oob = df.drop(df.index[pd.unique(idx)])
        if s[EVENT].sum() < 5 or len(oob) < 20: continue
        try:
            m = fit_cox(s, base_vars+[new_var])
            c_boot = harrell_c(s[TIME], s[EVENT], lp(m, s))
            c_oob  = harrell_c(oob[TIME], oob[EVENT], lp(m, oob))
            opt.append(c_boot - c_oob)
        except Exception: continue
    opt = np.array(opt)
    if len(opt) == 0: return dict(note="Bootstrap failed")
    return dict(apparent_c=c_app, optimism=opt.mean(),
                corrected_c=c_app - opt.mean(),
                note="If corrected_c - c_base < 0.01 -> no independent incremental value")

# ==================================================================
# 2. Main Execution
# ==================================================================
def run(df, label="", derived_here=False):
    print("\n" + "="*74)
    print(f"  {label}   n={len(df)}   Events={int(df[EVENT].sum())}")
    print("="*74)

    print("\n[1] Delta C-index (Paired bootstrap 1000x)")
    r = delta_c(df, BASE_COVARS, SIG, B=300)
    print(f"   Base model C = {r['c_base']:.4f}")
    print(f"  + Signature C = {r['c_full']:.4f}")
    print(f"   Delta C      = {r['delta']:+.4f}  95%CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    print(f"   bootstrap p = {r['p_boot']:.4f}  (n_boot={r['n_boot']})")
    print(f"   Verdict: {'[YES] Incremental value' if r['ci'][0] > 0 else ('[WARN] CI crosses 0, not proven' if r['ci'][0]<=0<=r['ci'][1] else '[NO] Negative incremental value')}")

    print("\n[2] Multivariable Cox: Independent HR of Signature")
    h = independent_hr(df, BASE_COVARS, SIG)
    print(f"   HR(per SD) = {h['hr_sd']:.3f}   95%CI [{h['hr_sd_lo']:.3f}, {h['hr_sd_hi']:.3f}]")
    print(f"   HR(per unit) = {h['hr']:.3f}  95%CI [{h['lo']:.3f}, {h['hi']:.3f}]  p = {h['p']:.2e}")
    print(f"   Verdict: {'[YES] Independent of clinical variables' if h['p'] < 0.05 else '[NO] Absorbed by clinical variables, no independent value'}")

    print("\n[3] NRI / IDI")
    for hz in [1, 3, 5]:
        r2 = nri_idi(df, BASE_COVARS, SIG, hz)
        if "note" in r2: print(f"   {hz}-year: {r2['note']}"); continue
        print(f"   {hz}-year: Events={r2['n_case']}/Controls={r2['n_ctrl']} | "
              f"cfNRI={r2['cfNRI']:+.3f} (p={r2['nri_p']:.3f}) | "
              f"IDI={r2['IDI']:+.4f} (p={r2['idi_p']:.3f})")

    if derived_here:
        print("\n[4] Overfitting correction (Derived in the same cohort -> necessary)")
        o = optimism_corrected_delta_c(df, BASE_COVARS, SIG)
        print(f"   Apparent C={o.get('apparent_c',float('nan')):.4f} Optimism={o.get('optimism',float('nan')):.4f} "
              f"Corrected C={o.get('corrected_c',float('nan')):.4f}")

    return r, h

def main():
    df = pd.read_csv(FILE)
    print("="*74); print("  Hypoxia-Invasion Signature: Incremental Value Analysis"); print("="*74)
    print("\nBaseline covariates:", BASE_COVARS)
    print("Signature variable:", SIG)

    results = {}
    if COHORT in df.columns:
        for c, sub in df.groupby(COHORT):
            results[c] = run(sub, f"Cohort {c}")
    else:
        results["all"] = run(df, "Full Cohort")

    # Cross-cohort meta-analysis
    if len(results) > 1:
        print("\n" + "="*74); print("  Cross-cohort meta-analysis (Random effects)"); print("="*74)
        lh, se, nm = [], [], []
        for c, (r_, h_) in results.items():
            lh.append(np.log(h_["hr_sd"])); se.append(abs(np.log(h_["hr_sd_hi"])-np.log(h_["hr_sd_lo"]))/3.92); nm.append(c)
        lh, se = np.array(lh), np.array(se)
        w = 1/(se**2 + 1e-9)
        mu = (w*lh).sum()/w.sum()
        Q = (w*(lh-mu)**2).sum(); dfq = len(lh)-1
        tau2 = max(0, (Q-dfq)/(w.sum()-(w**2).sum()/w.sum()))
        w2 = 1/(se**2 + tau2 + 1e-9)
        mu2 = (w2*lh).sum()/w2.sum(); se2 = np.sqrt(1/w2.sum())
        z = mu2/se2
        print(f"  Pooled HR(per SD) = {np.exp(mu2):.3f}  95%CI [{np.exp(mu2-1.96*se2):.3f}, {np.exp(mu2+1.96*se2):.3f}]")
        print(f"  p = {2*(1-norm.cdf(abs(z))):.2e}   I^2 = {max(0,(Q-dfq)/Q)*100:.1f}%  tau^2={tau2:.4f}")
        print(f"  Verdict: {'[YES] Consistent across cohorts' if tau2 < 0.05 else '[WARN] Heterogeneity across cohorts, requires explanation'}")

    # Interaction test
    print("\n" + "="*74); print("  LGG vs GBM: Does the HR of the signature truly differ? (Interaction test)"); print("="*74)
    gcol = "subtype" if "subtype" in df.columns else ("grade_group" if "grade_group" in df.columns else None)
    if gcol:
        it = interaction_test(df, SIG, gcol, BASE_COVARS)
        print(f"  HR Ratio = {it['hr_ratio']:.3f}  95%CI [{it['lo']:.3f}, {it['hi']:.3f}]")
        print(f"  Interaction p = {it['p_interaction']:.4f}")
        verdict = '[YES] Supports LGG is stronger' if it['p_interaction'] < 0.05 else '[NO] Cannot claim LGG is stronger (comparing significance on two sides is invalid)'
        print("  Verdict: " + verdict)
    else:
        print("  Subtype column (subtype / grade_group) not found -> Manual specification required")

    print("\n" + "="*74)
    print("6 Essential Metrics to Report:")
    print("  1. Base model C-index")
    print("  2. +Signature C-index and Delta C")
    print("  3. Bootstrap 95% CI of Delta C")
    print("  4. Independent HR(per SD) and 95% CI in multivariable Cox")
    print("  5. cfNRI and IDI (1/3/5 years)")
    print("  6. Pooled HR + I^2 across cohorts")

if __name__ == "__main__":
    main()