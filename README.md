# Glycemic Persistence Index (GPI)

Python code accompanying **CGM-Derived Glycemic Persistence Index Integrates Glucose Magnitude and Duration**.

GPI is the largest integer k for which glucose is ≥k mg/dL for a cumulative duration of ≥k minutes per day. Observations need not be consecutive.

## Programs

| File | Purpose |
|---|---|
| `compute_gpi.py` | Daily GPI and conventional CGM metrics |
| `reproduce_figure2a.py` | Figure 2A treatment trajectories |
| `analyze_dclp3.py` | ANCOVA and sensitivity analyses; Figure 2B within-TIR variation; Figure 2E day-to-day variability |
| `analyze_aireadi.py` | Figure 3 distributions; Figure 4 HbA1c associations and ROC analysis; Table 1 |

Scripts use cleaned inputs in the formats below. They generate analytical plots and numerical summaries rather than identical manuscript artwork. The calculator implements the GPI definition underlying Figure 1. Individual trace panels 2C–D, Table 2 subgroup correlations, and supplementary figures/tables are outside this package's scope.

## Installation and use

Python 3.12 is recommended.

```bash
python -m pip install -r requirements.txt
python compute_gpi.py --input examples/synthetic_cgm.csv --output daily_metrics.csv
python reproduce_figure2a.py --input examples/synthetic_daily_gpi.csv --output-dir outputs/figure2a --synthetic
python analyze_dclp3.py --input examples/synthetic_dclp3_daily.csv --output-dir outputs/dclp3 --synthetic
python analyze_aireadi.py --input examples/synthetic_aireadi_participants.csv --output-dir outputs/aireadi --synthetic
```

The example CSVs are entirely synthetic and demonstrate separate input formats; they do not reproduce study results. Replace the input paths with authorized data to run the study analyses and omit `--synthetic`. Use new output directories. Synthetic plots are labeled automatically when all identifiers start with `SYN_`.

## GPI calculator

Required columns: `participant_id,timestamp,glucose_mg_dl`.

Supply cleaned native five-minute observations with consistent calendar-day timestamps. Apply sensor validity rules and remove duplicate or conflicting timestamps upstream. The calculator does not resample, interpolate, or assign duration to missing observations. Unparseable timestamps, nonnumeric glucose, and missing identifiers are dropped; nonpositive or infinite glucose is invalid.

The importable function `compute_gpi(glucose_values, sampling_interval_minutes=5)` supports positive regular sampling intervals. The daily command uses five minutes per reading. Different sampling frequencies require corresponding completeness rules; equivalence across frequencies has not been established.

The generic calculator requires ≥200 readings/day; days below this threshold have missing metrics. For study analyses, apply the stricter cohort-specific criteria below. Daily GPI is an integer; retain full precision when averaging days. Calculator glucose CV is a fraction, whereas TIR/TAR/TBR are percentages.

## DCLP3

Figure 2A input columns:

`participant_id,treatment,phase,relative_day,n_readings,gpi`

The DCLP3 analysis additionally requires:

`date,mean_glucose,tir_70_180,tar_180_250,tar_gt_180,tar_gt_250`

Supply one complete row per participant/calendar day. Treatment is `CLC` or `SAP` and must remain constant for each participant. Phase is `1. Baseline` or `2. Post Randomization`. Relative day is calendar date minus randomization date, with randomization day 0. Reading counts and daily metrics must reflect the original cleaning and five-minute observations. Time metrics are percentages; mean glucose is mg/dL. TIR includes 70 and 180; TAR 180–250 means >180 and ≤250. No missing readings or days are imputed.

### Eligibility and calculations

- Primary days contain 202–288 valid readings. Baseline follows the source-designated phase, regardless of relative date. Follow-up includes post-randomization-phase days 1–182, excluding day 0.
- Participant means average qualifying daily values with equal day weights. ANCOVA includes participants with both baseline and follow-up means, with equal participant weights.
- Figure 2A follow-up intervals are days 1–28, 29–56, 57–84, 85–112, 113–140, and 141–182. Group means average available participant means; error bars are SEM (sample SD / sqrt(n)). The plot uses the manuscript's fixed GPI axis, 195–240.
- ANCOVA fits follow-up GPI ~ treatment + baseline GPI, with intercept. HC3 uses Student-t inference. The interaction model centers baseline at the pooled mean, so its treatment coefficient equals the predicted contrast averaged over the observed baseline distribution.
- Diagnostics include residual and normal Q–Q plots, Breusch–Pagan and Shapiro–Wilk tests. Completeness sensitivity repeats aggregation and ANCOVA at 231–288 and 260–288 readings/day.
- Figure 2B uses left-closed 5-percentage-point TIR bands; the final 95–100 band includes 100. Models include categorical TIR band alone and TIR band plus participant fixed effects. Residual SD is sqrt(SSE / residual degrees of freedom).
- Figure 2E calculates sample SD / mean across qualifying baseline days for each participant and metric, pooling treatment groups. At least two days and a positive mean are required; undefined CVs are excluded. CV is reported as a fraction.

Outputs include Figure 2A treatment and participant summaries, `ancova.json`, diagnostic plots, `within_tir.json`, and Figure 2B/2E summary CSVs and plots.

## AI-READI

Required columns, one row per participant:

`participant_id,study_group,hba1c,gpi,mean_glucose,tar_gt_180,tir_70_180,cv_glucose`

Prepare participant metrics as arithmetic means of daily metrics from days with ≥230 valid readings; no upper daily count limit was used. Each participant must have at least one qualifying day. Participant glucose CV is the average daily CV, not the CV of daily means. Use percentages for CV, TIR, TAR, and HbA1c, and mg/dL for mean glucose. Convert fractional time metrics and CV to percentages before supplying this input. Retain original cohort assignments rather than inferring groups from HbA1c.

Use these group labels:

- `Healthy`
- `Prediabetes/lifestyle-controlled`
- `Oral/non-insulin diabetes`
- `Insulin-treated diabetes`

The distribution cohort contains 1,898 participants; the HbA1c-matched cohort contains 1,848. Supply blank HbA1c where unavailable. Figure 3A restricts healthy participants to HbA1c <5.7%; Figure 3B uses all participants. Figure 4 and Table 1 exclude missing HbA1c and require complete comparator metrics in the same remaining participants. Comparator fields may be blank when HbA1c is absent. Exact seeded resampling requires the same participant order as the source analysis; input order is preserved.

### Statistical calculations

Figure 4A reports overall Pearson and Spearman correlations. ROC analyses evaluate HbA1c ≥5.7% and ≥6.5%. TIR is negated for ROC calculations; exported native TIR thresholds use ≤, while other thresholds use ≥.

Full-cohort cutoffs maximize the Youden index using `roc_curve(drop_intermediate=True)` and the first exact maximum among descending retained thresholds. AUC and cutoff percentile intervals use 2,000 unrestricted participant bootstrap draws (seed 264707), shared across metrics and endpoints. One-class draws are omitted; nonfinite cutoffs are omitted from cutoff intervals. Valid draw counts are exported.

Internal validation uses ten repeats of stratified five-fold cross-validation (seeds 7000–7009). Metrics share splits within each endpoint. Training-selected cutoffs are applied to held-out participants; classifications are pooled within each repeat before averaging performance. Table 1 AUCs are full-cohort estimates. Cross-validation evaluates cutoff selection, not external validity of the fixed full-cohort cutoffs. Predictive values depend on cohort prevalence.

Outputs include Figure 3/4 plots, distribution and overall correlation summaries, `table1.csv`, `cross_validation_by_repeat.csv`, and `training_cutoffs.csv`.

## Data access

The calculator and synthetic examples run without restricted data. Reproducing study results requires authorized DCLP3 or AI-READI data and preparation of the documented inputs using the same preprocessing, clinical linkage, and eligibility rules. Raw-data extraction and clinical linkage are not automated by these scripts.

No participant data or study outputs are distributed. Keep authorized inputs and derived participant-level outputs outside the public repository and follow the data providers' access and sharing terms.
