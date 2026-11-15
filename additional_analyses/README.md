# Additional analyses — scope and limitations

These scripts were used for sensitivity checks and audits described in the manuscript. They are provided **for transparency**. Note that these are not turnkey scripts: each reads a local input file whose path is defined in a variable near the top of the script. Please update this variable before running.

| Script | Purpose | Required input |
|---|---|---|
| `incremental_value_analysis.py` | Incremental prognostic value of the signature over established clinical/molecular covariates (delta C-index with paired bootstrap, independent HR, category-free NRI/IDI, cross-cohort meta-analysis, subtype interaction test) | `glioma_survival.csv` (set `FILE` at the top; also `TIME`, `EVENT`, `SIG`, `BASE_COVARS`) |
| `audit_and_fig1c.py` | Reconciles two versions of the tumour-level PAN–LE delta; converts a cell-level comparison to tumour-level pseudobulk; leave-one-out diagnostics | Per-tumour delta tables / single-cell metadata |
| `step_malignant_state_analysis.py` | Malignant-cell filtering verification, depth adjustment, Neftel cell-state and hypoxia scoring at tumour level | Single-cell metadata with cluster and patient columns |
| `step_deconfound_quantify.py` | Effect sizes with confidence intervals, attenuation test (delta_all minus delta_malignant), malignant-fraction association, state-axis analysis | Tumour-level summary tables |
| `step_decisive_positive_control.py` | Positive controls (hypoxia / mesenchymal / housekeeping gene sets) and within-tumour continuous state-axis quartile analysis | Expression matrix + gene sets |

### Important Notes

* These scripts were used for exploratory and sensitivity analyses. They are **not** the code that generated the manuscript's main figures; the main methodological figure is produced by `../pseudoreplication_benchmark.py`.
* Some scripts reference column names specific to the authors' local data tables. Users may need to adjust these column names according to their own data structures.
* Outputs from these additional analyses are not included in `../results/`. Please re-run them on the corresponding input data before citing any specific numeric values derived from them.

For the complete list of analyses that are **not** included in this deposit, see Section 6 of the top-level `README.md`.