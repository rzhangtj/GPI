#!/usr/bin/env python3
"""AI-READI distributions, HbA1c association, ROC curves, and Table 1 estimates."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import StratifiedKFold

METRICS=['gpi','mean_glucose','tar_gt_180','tir_70_180','cv_glucose']

def select_cutoff(y,x):
    """First maximum Youden index on descending, pruned ROC thresholds."""
    fpr,tpr,thresholds=roc_curve(y,x,drop_intermediate=True)
    return thresholds[int(np.argmax(tpr-fpr))]

def performance(y,p):
    tp=np.sum(y & p);tn=np.sum(~y & ~p);fp=np.sum(~y & p);fn=np.sum(y & ~p)
    def ratio(a,b):return float(a/b) if b else np.nan
    return dict(sensitivity=ratio(tp,tp+fn),specificity=ratio(tn,tn+fp),ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--synthetic',action='store_true')
    a=parser.parse_args()
    if a.bootstrap<1:raise ValueError('Bootstrap count must be positive.')
    d=pd.read_csv(a.input,dtype={'participant_id':str})
    required=['participant_id','study_group','hba1c',*METRICS]
    if set(required)-set(d):raise ValueError('Missing columns: '+str(sorted(set(required)-set(d))))
    if d.participant_id.isna().any() or d.participant_id.duplicated().any():raise ValueError('Unique nonmissing participant identifiers required.')
    if d.study_group.isna().any() or not np.isfinite(d.gpi.to_numpy(float)).all():raise ValueError('Group and GPI must be complete and finite.')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    label='Synthetic demonstration — ' if a.synthetic or d.participant_id.str.startswith('SYN_').all() else ''
    order=['Healthy','Prediabetes/lifestyle-controlled','Oral/non-insulin diabetes','Insulin-treated diabetes']
    if not d.study_group.isin(order).all():raise ValueError('Use the four study_group labels documented in README.')
    groups=[g for g in order if g in set(d.study_group)]
    d.groupby('study_group').gpi.agg(n='count',mean='mean',sd='std',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).to_csv(out/'figure3_summary.csv')
    reference=d.loc[d.study_group.eq('Healthy') & d.hba1c.lt(5.7),'gpi']
    if reference.empty:raise ValueError('Figure 3A requires healthy participants with HbA1c <5.7%.')
    pd.DataFrame([dict(n=len(reference),median=reference.median(),q25=reference.quantile(.25),q75=reference.quantile(.75))]).to_csv(out/'figure3a_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(reference,bins=20,color='#52768c')
    axes[0].set(xlabel='Participant GPI',ylabel='Participants',title='Healthy group, HbA1c <5.7%')
    axes[1].boxplot([d.loc[d.study_group.eq(g),'gpi'] for g in groups],tick_labels=groups,showfliers=False)
    axes[1].tick_params(axis='x',rotation=20);axes[1].set(ylabel='Participant GPI')
    fig.suptitle(label+'GPI distributions');fig.tight_layout();fig.savefig(out/'figure3.png',dpi=300);plt.close(fig)
    z=d.dropna(subset=['hba1c']).reset_index(drop=True)
    if not np.isfinite(z[['hba1c',*METRICS]].to_numpy(float)).all():raise ValueError('All metrics must be complete and finite in participants with HbA1c; use a common cohort.')
    pd.DataFrame([dict(n=len(z),pearson_r=pearsonr(z.gpi,z.hba1c).statistic,spearman_r=spearmanr(z.gpi,z.hba1c).statistic)]).to_csv(out/'figure4a_summary.csv',index=False)
    # Input order is deliberately preserved: seeded resampling depends on row order.
    rng=np.random.default_rng(264707)
    draws=[rng.integers(0,len(z),len(z)) for _ in range(a.bootstrap)]
    rows=[];cvrows=[];foldrows=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].scatter(z.gpi,z.hba1c,s=9,alpha=.4);axes[0].set(xlabel='Participant GPI',ylabel='HbA1c (%)')
    for endpoint in [5.7,6.5]:
        y=z.hba1c.ge(endpoint).to_numpy()
        if min(np.bincount(y.astype(int),minlength=2))<5:raise ValueError('Each endpoint requires at least five participants in each outcome class.')
        splits=[list(StratifiedKFold(5,shuffle=True,random_state=seed).split(z,y)) for seed in range(7000,7010)]
        for metric in METRICS:
            direction=-1 if metric=='tir_70_180' else 1
            x=z[metric].to_numpy()*direction
            cutoff=select_cutoff(y,x); auc=roc_auc_score(y,x)
            aucs=[];cuts=[]
            for ix in draws:
                if np.unique(y[ix]).size<2:continue
                aucs.append(roc_auc_score(y[ix],x[ix]));cuts.append(select_cutoff(y[ix],x[ix])*direction)
            repeats=[]
            for rep,split in enumerate(splits):
                pred=np.zeros(len(y),bool)
                for fold,(train,test) in enumerate(split):
                    c=select_cutoff(y[train],x[train]);pred[test]=x[test]>=c
                    foldrows.append(dict(endpoint=endpoint,metric=metric,repeat=rep,seed=7000+rep,fold=fold,cutoff=c*direction))
                perf=performance(y,pred);repeats.append(perf);cvrows.append(dict(endpoint=endpoint,metric=metric,repeat=rep,**perf))
            row=dict(endpoint=endpoint,metric=metric,n=len(z),n_positive=int(y.sum()),prevalence=float(y.mean()),auc=auc,
                auc_ci_low=np.percentile(aucs,2.5),auc_ci_high=np.percentile(aucs,97.5),
                full_cohort_cutoff=cutoff*direction,cutoff_operator='<=' if direction<0 else '>=',
                bootstrap_valid=len(aucs))
            finite=np.array(cuts)[np.isfinite(cuts)]
            row.update(cutoff_ci_low=np.percentile(finite,2.5) if len(finite) else np.nan,cutoff_ci_high=np.percentile(finite,97.5) if len(finite) else np.nan,cutoff_bootstrap_valid=len(finite))
            row.update({'apparent_'+k:v for k,v in performance(y,x>=cutoff).items()})
            row.update({'cv_'+k:float(np.nanmean([p[k] for p in repeats])) for k in repeats[0]})
            rows.append(row)
            if metric=='gpi':
                f,t,c=roc_curve(y,x,drop_intermediate=True)
                axes[1].plot(f,t,label=f'HbA1c ≥{endpoint}% (AUC {auc:.3f})')
                finite=np.isfinite(c);axes[2].plot(c[finite],(t-f)[finite],label=f'HbA1c ≥{endpoint}%')
                axes[2].axvline(cutoff,ls=':',alpha=.5)
    axes[1].plot([0,1],[0,1],ls='--',color='gray');axes[1].set(xlabel='1 − specificity',ylabel='Sensitivity');axes[1].legend(fontsize=8)
    axes[2].set(xlabel='GPI cutoff',ylabel='Youden index');axes[2].legend(fontsize=8)
    fig.suptitle(label+'HbA1c clinical anchoring');fig.tight_layout();fig.savefig(out/'figure4.png',dpi=300);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'table1.csv',index=False)
    pd.DataFrame(cvrows).to_csv(out/'cross_validation_by_repeat.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'training_cutoffs.csv',index=False)
    print('Analysis complete. Outputs:',out)

if __name__=='__main__':main()
