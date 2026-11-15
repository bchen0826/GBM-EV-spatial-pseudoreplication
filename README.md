# Code for: Spatial transcriptomic inference requires tumour-level aggregation

Monte Carlo benchmarks and supporting analysis scripts for the methodological
component of the manuscript on extracellular-vesicle biogenesis programmes in
IDH-wildtype glioblastoma.

---

## 1. Scope of this deposit — please read first

**This deposit contains the code for the methodological / simulation benchmark
and for the supporting sensitivity analyses. It is not a complete
end-to-end reproduction of the manuscript.**

Specifically, it contains:

* the Monte Carlo simulation that quantifies when spot-level inference fails
  (Figure 5: type I error inflation and power curves);
* a companion benchmark that re-expresses the same result on the axis of
  **residual Moran's I**, so that a real dataset can be located on the curve;
* the simulation outputs (CSV) and the figure produced by these scripts;
* auxiliary scripts used for sensitivity and audit analyses.

It does **not** contain (see Section 6) the single-cell clustering pipeline,
the 89-gene signature scoring wrapper, or the raw spatial transcriptomic
matrices. The *Code availability* statement in the manuscript must describe
exactly the scope above; no broader claim is made here.

---

## 2. Contents

| File | Purpose |
|---|---|
| `pseudoreplication_benchmark.py` | Main Monte Carlo benchmark. Separates **within-tumour** contrasts (PAN vs LE; inflation driven by spatial autocorrelation) from **between-tumour** contrasts (driven by ICC). Compares spot-level tests, tumour-level aggregation, and a random-intercept mixed model. Also computes power under between-tumour effect heterogeneity (tau). |
| `moran_type1_benchmark.py` | Companion benchmark on the residual Moran's I axis. Self-contained; reproduces `results/t1e_by_moran.csv`. |
| `requirements.txt` | Python dependencies. |
| `CHANGELOG.md` | Version history of the scoring/simulation rules. |
| `results/` | Simulation outputs and the three-panel figure. |
| `additional_analyses/` | Auxiliary scripts for sensitivity, audit and incremental-value analyses (see the README inside that folder). |
| `archive/pseudoreplication_benchmark_v2.0.py` | Earlier version of the main script, retained for transparency (without the Moran's I extension). |

### `results/`

| File | Content |
|---|---|
| `t1e_within.csv` | Type I error, within-tumour contrast, across spatial autocorrelation levels |
| `t1e_between.csv` | Type I error, between-tumour contrast, across ICC levels |
| `power_within_n9.csv` | Power curves for n = 9 tumours with effect heterogeneity tau = 0.30 |
| `t1e_by_moran.csv` | Type I error expressed against measured residual Moran's I |
| `pseudoreplication_benchmark.png` / `.pdf` | Three-panel summary figure |

> **Important — replicate counts differ between files.**
> `t1e_within.csv`, `t1e_between.csv`, `power_within_n9.csv` and the figure
> were produced by the main script in **quick mode** (B = 150 for type I error,
> B = 100 for power). `t1e_by_moran.csv` was produced by
> `moran_type1_benchmark.py` in **default mode** (B = 100 per level).
> For publication-grade numbers, re-run the main script without `--quick`
> (B = 400 / 250). Monte Carlo rates will shift slightly between runs; the
> qualitative conclusion is unchanged. **The CSV files in this deposit are the
> authoritative values for any number quoted in the manuscript** — do not mix
> them with rates from earlier exploratory runs.

---

## 3. System requirements

* Python 3.9 or newer
* Packages listed in `requirements.txt` — **including `statsmodels`**, which is
  required by the mixed-model comparison and is easy to overlook.

```bash
pip install -r requirements.txt
```

No R installation is needed to run the two benchmark scripts.

---

## 4. Reproducing

Run from a writable working directory; all outputs are written to the current
directory.

```bash
# Main benchmark (Figure 5)
python pseudoreplication_benchmark.py --quick      # smoke test, ~2 min
python pseudoreplication_benchmark.py             # full run, ~10–20 min

# Moran's I benchmark
python moran_type1_benchmark.py --quick
python moran_type1_benchmark.py
```

Outputs of the main script: `t1e_within.csv`, `t1e_between.csv`,
`power_within_n9.csv`, `pseudoreplication_benchmark.png`, `.pdf`.

### Optional: anchoring to real data

If a spot-level table with coordinates is available
(columns: `sample`, `compartment`, `EV_score`, `x`, `y`), the main script can
estimate residual spatial autocorrelation per tumour:

```bash
python pseudoreplication_benchmark.py --moran spots_with_coords.csv --xcol x --ycol y
```

This writes `moran_per_tumour.csv` and a calibration table. Residuals are
computed by subtracting the mean of each **tumour x compartment** stratum;
pooling spots across tumours would create artificial spatial structure.

*Note on calibration:* the `--moran` option maps Moran's I back to the
simulation smoothing parameter using an approximate Gaussian-field
calibration that saturates above rho ~ 0.3. For quoting type I error rates,
use `t1e_by_moran.csv` (Moran's I axis) as the primary reference; treat the
rho estimate as descriptive only.

---

## 5. Data availability

No newly generated experimental data are deposited here. The `results/` CSVs
are **simulation outputs**, not measurements.

Spatial transcriptomic and bulk cohorts used in the associated manuscript are
publicly available from their original repositories (GSE237183; Ravi et al.;
Ivy GAP; CGGA), with accession numbers and retrieval details given in the
manuscript Methods. The single-cell atlas is likewise from a public
repository; its accession is listed in the manuscript.

---

## 6. Not included in this deposit

The following were performed interactively in the authors' local environment
and are **not** provided as executable scripts here. Their parameters,
gene sets and decision rules are fully specified in the manuscript Methods so
that they can be reimplemented:

1. Single-cell quality control, clustering and cell-state annotation
   (Seurat/Scanpy pipeline).
2. The 89-gene EV-biogenesis signature scoring wrapper (gene set and scoring
   procedure described in Methods and supplementary tables).
3. Raw spatial transcriptomic matrices and spot-level count tables (public
   sources listed above).
4. CGGA preprocessing and the survival-model wrapper. An executable
   *incremental-value* analysis script is provided in `additional_analyses/`
   (`incremental_value_analysis.py`), but it requires a local
   `glioma_survival.csv` input.

If the journal requires end-to-end code, these four items must be added to a
new version of this deposit before resubmission; they cannot be recreated from
the files here.

---

## 7. Version and change log

See `CHANGELOG.md`. The current main script is **v2.1**; `archive/` contains
v2.0. Figure 5 panels 1–2 and the power curves are identical between v2.0 and
v2.1; v2.1 adds the Moran's I interface and the separate companion benchmark.

---

## 8. Licence

Figshare requires a licence field for DOI minting. Note that **Creative Commons
licences are not appropriate for software/code**; a permissive software
licence (for example MIT or BSD-3-Clause), or the licence required by the
funder or institution, should be selected for the code, with any different
terms for the accompanying documentation stated in the item description.

---

## 9. Citation

Cite this deposit using the DOI issued by the repository (after reservation
and publication), for example as:

> Chen J. *Code for: Spatial transcriptomic inference requires tumour-level
> aggregation*. Figshare. DOI: [repository-issued DOI].

Add the paper DOI to the item's *Related materials* field with the
`IsSupplementTo` relationship once the article is published.
