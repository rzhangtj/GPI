#!/usr/bin/env python3
"""Participant-level ANCOVA, within-TIR variation, and baseline stability."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

METRICS = ['gpi', 'mean_glucose', 'tir_70_180', 'tar_180_250', 'tar_gt_180', 'tar_gt_250']

def select_days(data, minimum=202):
    q = data[data.n_readings.between(minimum, 288)].copy()
    baseline = q[q.phase.eq('1. Baseline')].copy()
    followup = q[q.phase.eq('2. Post Randomization') & q.relative_day.between(1, 182)].copy()
    return baseline, followup

def participants(baseline, followup):
    b = baseline.groupby('participant_id').gpi.mean().rename('baseline')
    p = followup.groupby('participant_id').gpi.mean().rename('post')
    arms = baseline.groupby('participant_id').treatment.first()
    s = pd.concat([b, p, arms], axis=1).dropna()
    s['treat'] = s.treatment.eq('CLC').astype(int)
    return s

def ancova(s):
    if len(s) < 6 or s.treat.nunique() != 2:
        raise ValueError('ANCOVA requires both treatment groups and sufficient participants.')
    model = smf.ols('post ~ treat + baseline', s).fit()
    robust = model.get_robustcov_results(cov_type='HC3', use_t=True)
    j = model.model.exog_names.index('treat')
    s = s.copy()
    s['centered_baseline'] = s.baseline - s.baseline.mean()
    interaction = smf.ols('post ~ treat * centered_baseline', s).fit(cov_type='HC3', use_t=True)
    result = dict(n=len(s), n_closed_loop=int(s.treat.sum()),
        baseline_mean=float(s.baseline.mean()), followup_mean=float(s.post.mean()),
        contrast=float(model.params['treat']), ordinary_ci=model.conf_int().loc['treat'].tolist(),
        ordinary_p=float(model.pvalues['treat']), hc3_ci=robust.conf_int()[j].tolist(),
        hc3_p=float(robust.pvalues[j]),
        breusch_pagan_p=float(het_breuschpagan(model.resid, model.model.exog)[1]),
        shapiro_wilk_p=float(shapiro(model.resid).pvalue),
        interaction_p_hc3=float(interaction.pvalues['treat:centered_baseline']),
        average_interaction_contrast=float(interaction.params['treat']),
        average_interaction_ci_hc3=interaction.conf_int().loc['treat'].tolist())
    for arm, value in [('closed_loop', 1), ('control', 0)]:
        result['adjusted_mean_' + arm] = float(model.predict(pd.DataFrame({'treat':[value], 'baseline':[s.baseline.mean()]})).iloc[0])
    return result, model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--synthetic', action='store_true')
    a = parser.parse_args()
    d = pd.read_csv(a.input, dtype={'participant_id':str})
    required = ['participant_id','date','treatment','phase','relative_day','n_readings',*METRICS]
    if set(required)-set(d): raise ValueError('Missing columns: '+str(sorted(set(required)-set(d))))
    if d[required].isna().any().any(): raise ValueError('Required fields must be complete.')
    if d.duplicated(['participant_id','date']).any(): raise ValueError('Duplicate participant dates.')
    if not d.treatment.isin(['CLC','SAP']).all(): raise ValueError('Use CLC or SAP treatment labels.')
    if d.groupby('participant_id').treatment.nunique().max()>1: raise ValueError('Inconsistent treatment assignment.')
    numeric=['relative_day','n_readings',*METRICS]
    if not np.isfinite(d[numeric].to_numpy(float)).all(): raise ValueError('Nonfinite numerical inputs.')
    if (d.n_readings%1 != 0).any() or (d.relative_day%1 != 0).any(): raise ValueError('Day offsets and reading counts must be integers.')
    if not d[METRICS[2:]].ge(0).all().all() or not d[METRICS[2:]].le(100).all().all(): raise ValueError('Time metrics must be percentages from 0 to 100.')
    out=Path(a.output_dir);out.mkdir(parents=True, exist_ok=False)
    label='Synthetic demonstration — ' if a.synthetic or d.participant_id.str.startswith('SYN_').all() else ''
    results=[]
    for minimum in [202,231,260]:
        b,p=select_days(d,minimum);s=participants(b,p)
        result,model=ancova(s)
        result.update(minimum_readings=minimum, baseline_days=len(b), followup_days=len(p))
        results.append(result)
        if minimum==202:
            primary_b,primary_p=b,p
            fig,axes=plt.subplots(1,2,figsize=(9,3.5))
            axes[0].scatter(model.fittedvalues,model.resid,s=10)
            axes[0].axhline(0,color='black',lw=1)
            axes[0].set(xlabel='Fitted GPI',ylabel='Residual')
            from scipy.stats import probplot
            probplot(model.resid,plot=axes[1]);fig.suptitle(label+'ANCOVA diagnostics')
            fig.tight_layout();fig.savefig(out/'ancova_diagnostics.png',dpi=200);plt.close(fig)
    (out/'ancova.json').write_text(json.dumps(results,indent=2)+'\n')
    p=primary_p.copy();p['tir_band']=np.minimum((p.tir_70_180//5).astype(int)*5,95)
    m1=smf.ols('gpi ~ C(tir_band)',p).fit()
    m2=smf.ols('gpi ~ C(tir_band) + C(participant_id)',p).fit()
    (out/'within_tir.json').write_text(json.dumps(dict(n_days=len(p),n_participants=p.participant_id.nunique(),
        r2_band=m1.rsquared,r2_band_and_participant=m2.rsquared,residual_sd=float(np.sqrt(m2.mse_resid))),indent=2)+'\n')
    bands=p.groupby('tir_band').gpi.agg(['count','mean','median','min','max'])
    bands.to_csv(out/'figure2b_summary.csv')
    fig,ax=plt.subplots(figsize=(9,4));groups=list(p.groupby('tir_band'))
    ax.boxplot([g.gpi for _,g in groups],tick_labels=[f'{k}–{k+5}' for k,_ in groups],showfliers=False)
    ax.tick_params(axis='x',rotation=60);ax.set(xlabel='TIR (%)',ylabel='Daily GPI',title=label+'Within-TIR distribution')
    fig.tight_layout();fig.savefig(out/'figure2b.png',dpi=300);plt.close(fig)
    values=[];rows=[]
    for metric in METRICS:
        z=primary_b.groupby('participant_id')[metric].agg(['mean','std','count'])
        cv=(z['std']/z['mean']).where((z['count']>=2)&(z['mean']>0)).dropna()
        values.append(cv)
        rows.append(dict(metric=metric,n=len(cv),median=cv.median(),q25=cv.quantile(.25),q75=cv.quantile(.75)))
    pd.DataFrame(rows).to_csv(out/'figure2e_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));ax.boxplot(values,tick_labels=['GPI','Mean glucose','TIR','TAR 180–250','TAR >180','TAR >250'],showfliers=False)
    ax.tick_params(axis='x',rotation=25);ax.set(ylabel='Within-participant CV (fraction)',title=label+'Baseline day-to-day variability')
    fig.tight_layout();fig.savefig(out/'figure2e.png',dpi=300);plt.close(fig)
    print('Analysis complete. Outputs:',out)

if __name__=='__main__':main()
