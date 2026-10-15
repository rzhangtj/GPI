# Glycemic Persistence Index (GPI)

Python code for calculating daily GPI from five-minute continuous glucose monitoring (CGM) observations and recreating Figure 2A treatment-response summaries from locally supplied daily GPI data.

GPI is the largest integer k for which glucose is at least k mg/dL for at least k represented minutes. It uses glucose in mg/dL and duration in minutes; it does not require a predefined glucose threshold. Contributing observations need not be consecutive.

## Contents

| File | Purpose |
|---|---|
| `compute_gpi.py` | Daily GPI and conventional CGM summary metrics. |
| `reproduce_figure2a.py` | Participant-interval aggregation, treatment means, SEM, and Figure 2A plotting. |
| `examples/synthetic_cgm.csv` | Entirely synthetic five-minute CGM observations. |
| `examples/synthetic_daily_gpi.csv` | Separate synthetic demonstration of Figure 2A input. |
| `requirements.txt` | Dependency versions used to check both examples. |

Both example datasets are synthetic. Neither contains actual participant records, and neither reproduces the published numerical results. They demonstrate separate workflows; the calculator example is not the source of the Figure 2A example.

## Installation

Python 3.12 is recommended. The examples were checked with Python 3.12.14 on Linux and the dependency versions listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

Run the commands below from this repository's root directory.

## 1. Calculate daily GPI

```bash
python compute_gpi.py --input examples/synthetic_cgm.csv --output synthetic_daily_metrics.csv
```

Input columns are `participant_id`, `timestamp`, and `glucose_mg_dl`. Supply preprocessed native five-minute observations, with glucose in mg/dL and consistent timestamps defining the intended calendar day. Each reading represents five minutes. The script does not infer sampling intervals, resample, fill missing observations, remove duplicate timestamps, or perform study-specific sensor validity checks. Clean these upstream; do not supply irregular or differently sampled data without adapting the represented-time calculation.

Unparseable timestamps or glucose values and missing identifiers are dropped. The calculator groups by participant and calendar date. Its existing default is at least **200 valid readings/day**; days below that threshold remain in the output with missing metrics. This generic calculator rule is distinct from the Figure 2A rule of **202–288 readings/day**. Passing calculator QC alone does not establish eligibility for Figure 2A.

Outputs include `gpi`, `n_readings`, `qc_pass`, mean and sample SD of glucose (mg/dL), glucose CV as a **fraction** (multiply by 100 for percent), and TIR/TAR/TBR as percentages. The function `compute_gpi` can also be imported for use with preprocessed glucose arrays. Its sampling interval argument defaults to five minutes.

## 2. Recreate Figure 2A

### Input and inclusion assumptions

Supply a CSV with one row per participant and recorded calendar study day:

| Column | Meaning |
|---|---|
| `participant_id` | Local participant identifier; consistent across days. |
| `treatment` | Randomized assignment: `CLC` = closed loop; `SAP` = control. |
| `phase` | Source-designated `1. Baseline` or `2. Post Randomization`. |
| `relative_day` | Integer recorded calendar date minus recorded randomization date, after normalizing both to midnight. Randomization is day 0. |
| `n_readings` | Number of valid native five-minute observations used for that day's GPI, after the original preprocessing. |
| `gpi` | Daily integer GPI under the manuscript's mg/dL-and-minutes convention; do not round participant means in advance. |

Daily values and counts must already reflect the original glucose-validity, duplicate, source-clock and daily-GPI processing. This script cannot verify those upstream steps. It applies the final **202–288 readings/day, inclusive** rule, so inputs may contain either only qualifying days or additional observed days with complete daily values and counts.

Baseline includes qualifying days in the **source-designated baseline phase**, regardless of relative day; do not redefine it using negative days. Follow-up includes qualifying post-randomization-phase days **1–182**, inclusive; day 0 is excluded. Follow-up intervals are 1–28, 29–56, 57–84, 85–112, 113–140 and 141–182, labelled 1–4, 5–8, 9–12, 13–16, 17–20 and 21–26 weeks. No days are filled or imputed.

Within each participant and interval, daily GPI is averaged arithmetically. Treatment means then average participant means with **equal participant weight**, regardless of day count. Error bars are **SEM = sample SD of participant means / sqrt(n)**, using SD with `ddof=1`. Counts reflect participants actually contributing to each point; no complete-case exclusion across intervals is imposed. SEM is undefined for fewer than two contributing participants. These are unadjusted means, not ANCOVA estimates. Plot axes and treatment colours follow the revised Figure 2A.

### Synthetic example

Run from the repository root:

```bash
python reproduce_figure2a.py --input examples/synthetic_daily_gpi.csv --output-dir outputs/figure2a_demo --synthetic
```

The small deterministic dataset uses clearly synthetic identifiers and relative days, covers both arms and all intervals, and includes a few excluded days to demonstrate the day-count and follow-up boundaries. `--synthetic` visibly labels the plot; synthetic identifiers beginning with `SYN_` also trigger this label automatically. This example is not evidence about treatment effects.

### Authorized local input

Prepare the same six columns from an authorized daily GPI dataset using the unchanged source phases, randomized assignments, recorded-date convention and preprocessing. Do not infer baseline from relative day. Run:

```bash
python reproduce_figure2a.py --input local_daily_gpi.csv --output-dir private/local_figure2a
```

Outputs are `figure2a.png`, `treatment_summary.csv` and `participant_interval_means.csv`; use a new output directory to preserve previous results. Keep authorized inputs and all derived outputs private, especially participant summaries. **Actual DCLP3 data are not distributed. Reproducing the published numerical results requires authorized access to those data.**

## Connecting the workflows

For authorized data, calculate daily GPI after study-specific preprocessing, then join daily results to randomized treatment assignment, source phase, and randomization date to prepare the six Figure 2A input columns. These metadata are not inferred by the calculator. Remove records without complete daily values before passing them to the Figure 2A script. Retain the original preprocessing and calendar conventions when reproducing the study.

## Data access and scope

Actual DCLP3 and AI-READI participant data are not included. Reproduction of the published Figure 2A numerical results requires authorized DCLP3-derived input. This repository does not reproduce all manuscript analyses or the original raw-CGM preprocessing. The Figure 2A y-axis is fixed to match the supplied figure code and may not suit other datasets.

Keep authorized data and derived participant-level outputs outside the public repository. The ignored `private/` and `outputs/` directories are conveniences, not substitutes for reviewing files before publication.
