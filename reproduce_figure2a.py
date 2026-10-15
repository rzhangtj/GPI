#!/usr/bin/env python3
"""Reproduce Figure 2A aggregation and plotting from locally supplied daily GPI."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

INTERVALS = ['Baseline', '1-4', '5-8', '9-12', '13-16', '17-20', '21-26']
EDGES = [0, 28, 56, 84, 112, 140, 182]
ARMS = [('CLC', '#cc0000', 'Closed loop'), ('SAP', '#0879b9', 'Control')]
BASELINE = '1. Baseline'
FOLLOWUP = '2. Post Randomization'
COLUMNS = ['participant_id', 'treatment', 'phase', 'relative_day', 'n_readings', 'gpi']


def read_daily(path):
    data = pd.read_csv(path, dtype={'participant_id': str})
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    data = data[COLUMNS].copy()
    if data.empty or data.isna().any().any():
        raise ValueError('Input must contain complete daily records.')
    if data.participant_id.str.strip().eq('').any():
        raise ValueError('Participant identifiers must not be empty.')
    for column in ['relative_day', 'n_readings', 'gpi']:
        data[column] = pd.to_numeric(data[column], errors='raise')
        if not np.isfinite(data[column]).all():
            raise ValueError(column + ' must be finite.')
    for column in ['relative_day', 'n_readings']:
        if not data[column].eq(np.floor(data[column])).all():
            raise ValueError(column + ' must contain integers.')
    if (data.n_readings < 0).any() or (data.gpi < 0).any():
        raise ValueError('Reading counts and daily GPI must be nonnegative.')
    if not data.treatment.isin(['CLC', 'SAP']).all():
        raise ValueError('Treatment must be randomized assignment CLC or SAP.')
    if not data.phase.isin([BASELINE, FOLLOWUP]).all():
        raise ValueError('Phase must be 1. Baseline or 2. Post Randomization.')
    if data.groupby('participant_id').treatment.nunique().gt(1).any():
        raise ValueError('A participant must have one randomized treatment assignment.')
    if data.duplicated(['participant_id', 'relative_day']).any():
        raise ValueError('Provide exactly one row per participant/calendar study day.')
    return data


def summarize(data):
    # The source phase defines baseline; do not replace it with negative days.
    eligible = data[data.n_readings.between(202, 288)].copy()
    eligible['interval'] = pd.cut(
        eligible.relative_day, EDGES, labels=INTERVALS[1:], right=True
    ).astype(object)
    eligible.loc[eligible.phase.eq(BASELINE), 'interval'] = 'Baseline'
    eligible = eligible[eligible.interval.notna()].copy()
    if eligible.empty:
        raise ValueError('No eligible baseline or follow-up days.')
    # First average days per person and interval, then weight people equally.
    participants = eligible.groupby(
        ['participant_id', 'treatment', 'interval'], sort=True
    ).agg(mean_daily_gpi=('gpi', 'mean'), qualifying_days=('gpi', 'size')).reset_index()
    summary = participants.groupby(['treatment', 'interval'], sort=True).agg(
        mean=('mean_daily_gpi', 'mean'), sd=('mean_daily_gpi', 'std'),
        participant_count=('mean_daily_gpi', 'size'),
        qualifying_day_count=('qualifying_days', 'sum')
    ).reset_index()
    summary['sem'] = summary.sd / np.sqrt(summary.participant_count)
    summary['treatment_label'] = summary.treatment.map({'CLC': 'Closed loop', 'SAP': 'Control'})
    return eligible, participants, summary


def plot(summary, path, synthetic=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(6.3, 5.1))
    handles = []
    for arm, color, label in ARMS:
        rows = summary[summary.treatment.eq(arm)].set_index('interval').reindex(INTERVALS)
        counts = rows.participant_count.to_numpy()
        legend = f'{label} (N={int(counts[0])})' if np.isfinite(counts).all() and np.all(counts == counts[0]) else label
        ax.errorbar(np.arange(7), rows['mean'], yerr=rows['sem'], fmt='o-',
                    color=color, linewidth=2, markersize=7, ecolor='#777777',
                    elinewidth=1.5, capsize=4, capthick=1.5)
        handles.append(Line2D([], [], linestyle='-', linewidth=2, color=color,
                              marker='o', markersize=7, label=legend))
    ax.set_xticks(np.arange(7), INTERVALS, rotation=45, ha='right')
    ax.set_xlabel('Weeks', fontweight='bold', fontsize=14, labelpad=9)
    ax.set_ylabel('GPI', fontweight='bold', fontsize=14, labelpad=8)
    ax.set_ylim(195, 240)
    ax.set_yticks(np.arange(195, 241, 5))
    ax.set_xlim(-.35, 6.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.tick_params(direction='out', width=1.2, length=4)
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=11)
    if synthetic:
        fig.suptitle('SYNTHETIC EXAMPLE - not published results', fontsize=11, fontweight='bold', y=.985)
    fig.subplots_adjust(left=.18, bottom=.25, right=.98, top=.90 if synthetic else .94)
    fig.savefig(path, dpi=600, facecolor='white', metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Local daily GPI CSV.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Local directory for generated results.')
    parser.add_argument('--synthetic', action='store_true', help='Visibly mark a synthetic demonstration plot.')
    args = parser.parse_args()
    data = read_daily(args.input)
    _, participants, summary = summarize(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in ['participant_interval_means.csv', 'treatment_summary.csv', 'figure2a.png']]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Output files already exist; choose a new output directory.')
    participants.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    synthetic = args.synthetic or data.participant_id.str.startswith('SYN_').all()
    plot(summary, outputs[2], synthetic=synthetic)


if __name__ == '__main__':
    main()
