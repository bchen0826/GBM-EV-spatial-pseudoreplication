# Change log — simulation and scoring rules

This file records changes to the benchmark specification so that every
reported number can be traced to a specific version.

## v2.1 (current — `pseudoreplication_benchmark.py`)

* Added `--moran`, `--xcol`, `--ycol` options: per-tumour residual Moran's I
  estimation from a spot-level table with coordinates.
* Added `moran_type1_benchmark.py`, which expresses the type I error
  benchmark directly on the residual Moran's I axis (`t1e_by_moran.csv`).
  Rationale: the simulation parameter rho is not measurable in real data,
  whereas Moran's I is.
* Retained all v2.0 scenarios unchanged. Figure 5 panels 1–2 and the power
  curves are identical between v2.0 and v2.1.
* Known limitation: the rho calibration used by `--moran` saturates above
  rho ~ 0.3. Use `t1e_by_moran.csv` as the primary reference for quoting
  error rates.

## v2.0 (`archive/pseudoreplication_benchmark_v2.0.py`)

* Introduced the distinction between within-tumour and between-tumour
  contrasts. Finding: for within-tumour contrasts the tumour random intercept
  cancels on differencing, so ICC is not the source of inflation; spatial
  autocorrelation is. For between-tumour contrasts, ICC drives inflation
  (Kish design effect).
* Added the random-intercept mixed model as a third inference strategy.
  Finding: with spatially autocorrelated residuals, the mixed model fails in
  almost the same way as a spot-level test, because it assumes independent
  residuals.
* Added between-tumour effect heterogeneity (tau_delta) to power simulations.

## v1.0 (not archived)

* Single descriptive estimate of nominal inflation
  (observations / independent units). Superseded: the nominal ratio is an
  upper bound, not a design effect, and does not apply to within-tumour
  contrasts. v1.0 outputs are deliberately excluded from this deposit to
  avoid mixing incompatible specifications.

---

### Reporting convention

All type I error rates are empirical rejection frequencies at nominal
alpha = 0.05 under a true null effect. `B` (number of Monte Carlo replicates)
is recorded in each output CSV.

| Script | Quick | Default |
|---|---|---|
| `pseudoreplication_benchmark.py` | B = 150 (type I error), 100 (power) | B = 400 (type I error), 250 (power) |
| `moran_type1_benchmark.py` | B = 25 | B = 100 |

The CSVs shipped in `results/` reflect one quick-mode run of the main script
and one default-mode run of the companion script. Because Monte Carlo rates
vary between runs, always quote the shipped CSVs rather than values from
earlier exploratory sessions.
