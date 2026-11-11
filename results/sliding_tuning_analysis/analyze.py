"""Read saved experiment artifacts only; never load data or fit models."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
APPROACHES = ['baseline', 'tuned_fixed_threshold', 'tuned_optimized_threshold']
METRICS = ['f1', 'precision', 'recall', 'pr_auc', 'roc_auc']
BENCH = .7397

def table(df):
    df = df.copy()
    for c in df.select_dtypes('number'):
        df[c] = df[c].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    return '| ' + ' | '.join(df.columns) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(str(v).replace('|','/') for v in row) + ' |' for row in df.itertuples(index=False, name=None))

def main():
    dirs = sorted(p for p in (ROOT/'results').rglob('tuning_2026-11-11_*') if p.is_dir() and '21-33' <= p.name.split('2026-11-11_')[1][:5] <= '22-13')
    assert len(dirs) == 16, len(dirs)
    rows, folds, trials_out, configs, manifest = [], [], [], {}, []
    issues = []
    for p in dirs:
        texts = {}
        for name in ['tuning_summary.txt','comparison_summary.csv','outer_results.csv','trials.csv']:
            data = (p/name).read_bytes()
            manifest.append(dict(path=str((p/name).relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest()))
        txt = (p/'tuning_summary.txt').read_text(encoding='utf-8')
        cfg = ast.literal_eval(txt.split('FULL CONFIGURATION')[1].split('\n',2)[2].split('\n\nMETHODOLOGY NOTES')[0].strip())
        configs[p.name] = cfg
        model = cfg['hyperparameter_tuning']['model']
        window = cfg['validation']['sliding_window']['train_years']
        f = pd.read_csv(p/'outer_results.csv')
        t = pd.read_csv(p/'trials.csv')
        s = pd.read_csv(p/'comparison_summary.csv')
        assert set(f.model)=={model}
        assert set(f.outer_year)=={2020,2021,2022,2023}
        assert len(f)==12 and not f.duplicated(['outer_year','experiment']).any()
        assert set(f.experiment)==set(APPROACHES)
        assert not f[METRICS+['threshold','inner_folds']].isna().any().any()
        assert f[METRICS].ge(0).all().all() and f[METRICS].le(1).all().all()
        counts = t.groupby('outer_year').size()
        assert set(counts.index)=={2020,2021,2022,2023} and counts.eq(15).all()
        assert not t.duplicated(['outer_year','trial']).any()
        assert t.inner_f1_mean.notna().all()
        for a in APPROACHES:
            q=f[f.experiment.eq(a)]
            original=s[s.experiment.eq(a)].iloc[0]
            r=dict(directory=p.name,model=model,configured_window_years=window,outer_strategy_evidence='expanding_window_in_current_code',approach=a,configured_trials_per_fold=cfg['hyperparameter_tuning']['optuna']['n_trials'],actual_trials_total=len(t),actual_trials_per_fold=json.dumps(counts.to_dict()),actual_inner_folds=json.dumps(q.set_index('outer_year').inner_folds.to_dict()))
            for m in METRICS:
                for stat in ['mean','std']:
                    val=getattr(q[m],stat)()
                    assert np.isclose(val,original[f'{m}_{stat}'],atol=1e-12)
                    r[f'{m}_{stat}']=val
            r['delta_f1_vs_historical_rounded']=r['f1_mean']-BENCH
            rows.append(r)
        for year,g in f.groupby('outer_year'):
            best=t[t.outer_year.eq(year)].inner_f1_mean.max()
            assert np.isclose(best,g[g.experiment.eq('tuned_fixed_threshold')].inner_f1.iloc[0])
        for frame in [f,t]:
            frame.insert(0,'directory',p.name)
            frame.insert(1,'configured_window_years',window)
        folds.append(f)
        t['model']=model
        trials_out.append(t)
    s=pd.DataFrame(rows)
    f=pd.concat(folds,ignore_index=True)
    t=pd.concat(trials_out,ignore_index=True)
    assert s[['model','configured_window_years']].drop_duplicates().shape[0]==16
    s.to_csv(OUT/'all_experiments.csv',index=False)
    ranking=s.copy()
    ranking['rank_within_approach']=ranking.groupby('approach').f1_mean.rank(method='min',ascending=False).astype(int)
    ranking=ranking.sort_values(['approach','rank_within_approach','model'])
    ranking.to_csv(OUT/'model_ranking.csv',index=False)
    keys=['model','experiment','outer_year']
    pair=f[f.configured_window_years.eq(2)].merge(f[f.configured_window_years.eq(3)],on=keys,suffixes=('_2y','_3y'),validate='one_to_one')
    for m in METRICS:
        pair[f'delta_{m}_3y_minus_2y']=pair[f'{m}_3y']-pair[f'{m}_2y']
    pair.to_csv(OUT/'fold_comparison.csv',index=False)
    baseline_pairs=pair[pair.experiment.eq('baseline')]
    for m in METRICS+['train_rows','validation_rows']:
        assert np.allclose(baseline_pairs[f'{m}_2y'],baseline_pairs[f'{m}_3y'],atol=1e-12)
    f.to_csv(OUT/'outer_fold_details.csv',index=False)
    trial_stats=[]
    for (d,y),g in t.groupby(['directory','outer_year']):
        q=f[(f.directory==d)&(f.outer_year==y)&(f.experiment=='tuned_fixed_threshold')].iloc[0]
        z=f[(f.directory==d)&(f.outer_year==y)&(f.experiment=='tuned_optimized_threshold')].iloc[0]
        scores=g.inner_f1_mean
        trial_stats.append(dict(directory=d,model=q.model,configured_window_years=q.configured_window_years,outer_year=y,trials=len(g),unique_parameter_sets=g.parameters.map(lambda p:json.dumps(ast.literal_eval(p),sort_keys=True)).nunique(),first_trial_f1=scores.iloc[0],median_inner_f1=scores.median(),min_inner_f1=scores.min(),max_inner_f1=scores.max(),best_trial=int(g.loc[scores.idxmax(),'trial']),best_first10=g[g.trial<=10].inner_f1_mean.max(),best_last5=g[g.trial>10].inner_f1_mean.max(),inner_minus_outer_fixed=q.inner_f1-q.f1,threshold_inner_gain=z.threshold_inner_f1-q.inner_f1,outer_threshold_gain=z.f1-q.f1))
    ts=pd.DataFrame(trial_stats)
    ts.to_csv(OUT/'trial_diagnostics.csv',index=False)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report(s,ranking,f,pair,ts,configs)
    print(ranking[['model','configured_window_years','approach','f1_mean','pr_auc_mean','roc_auc_mean']].to_string(index=False))

def report(s,ranking,f,pair,ts,configs):
    blocks=['# Saved tuning results do not establish a sliding-window winner',
    '## Executive summary',
    'All 16 requested runs are present and complete (8 models × 2 configured windows; 192 outer metric rows, 960 trial rows). However, the current tuning implementation omits the outer strategy and window arguments, defaulting to expanding-window training. The saved training counts and identical baseline results across configured windows support this concern. These results compare configured inner-window selection procedures on apparently expanding outer training sets; they cannot establish that a 2- or 3-year outer sliding window is preferable.',
    'The historical reference is the user-supplied, rounded F1 of 0.7397. Small numerical exceedances are descriptive, not evidence of a reliable improvement. The strongest observed configurations below are exploratory candidates, subject to selection bias across 16 runs and three approaches.',
    '## Three separate rankings']
    for a in APPROACHES:
        q=ranking[ranking.approach.eq(a)]
        blocks += [f'### {a}',table(q[['rank_within_approach','model','configured_window_years','f1_mean','f1_std','precision_mean','recall_mean','pr_auc_mean','roc_auc_mean','delta_f1_vs_historical_rounded']])]
    # comparison chart, with separate panels and explicit configured labels
    fig,axes=plt.subplots(1,3,figsize=(17,7),sharex=True)
    labels=sorted(s.model.unique())
    for ax,a in zip(axes,APPROACHES):
        for w,marker in [(2,'o'),(3,'x')]:
            q=s[(s.approach==a)&(s.configured_window_years==w)].set_index('model').loc[labels]
            ax.scatter(q.f1_mean,np.arange(8)+(w-2)*.16,label=f'Configured {w} years',marker=marker)
        ax.axvline(BENCH,color='black',linestyle='--',label='Historical 0.7397')
        ax.set_yticks(np.arange(8)+.08,labels)
        ax.set_title(a.replace('_','\n')); ax.set_xlabel('Mean outer F1'); ax.grid(axis='x',alpha=.25)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Configured windows; outer strategy appears expanding in current implementation')
    fig.tight_layout(); fig.savefig(OUT/'f1_comparison.png',dpi=160); plt.close(fig)
    blocks += ['![Mean F1 by approach and configured window](f1_comparison.png)',
    '## Integrity and temporal interpretation',
    'Discovery used directory time only to bound the requested batch (21:33–22:13 on 11 November 2026). Model identity and window duration were extracted from each saved FULL CONFIGURATION and reconciled with CSV model names. All 16 model/window pairs occur once. All runs contain all four requested files, four outer years, three approaches per year and 15 unique trial IDs per year. No incomplete or duplicate experiment keys were found in this batch. Earlier tuning directories are outside scope, including incomplete earlier runs; they were not substituted into the batch.',
    'Current `src/tune_models.py:462` calls `temporal_cv(df, years, horizon)` without `strategy` or `train_years`. `src/validation.py:64` defaults to expanding windows. The inner-fold builder does read the configured sliding-window duration (unless explicitly overridden). Source snapshots from execution time are absent, so current code is corroborating evidence rather than a definitive historical execution trace. Training-row counts below are identical across all models and configured windows; baseline metric rows are also identical between windows for every model.',
    table(f[['outer_year','train_rows','validation_rows','inner_folds']].drop_duplicates().sort_values('outer_year')),
    'The saved configurations retain a 90-day target-label gap and years 2020–2023. The current split code enforces label availability strictly before validation starts. This analysis reads result files and source code only: it does not load raw datasets, access/evaluate 2024, train models, rerun Optuna, or alter the pipeline/configuration.',
    '## Configured 3-year minus 2-year comparisons',
    'Deltas below are in metric units (multiply by 100 for percentage points). They describe saved procedures, not a verified outer sliding-window contrast. PR-AUC here is average precision in the tuning implementation. Fixed and optimized tuned approaches share probabilities, so their PR-AUC and ROC-AUC are identical.']
    for a in APPROACHES:
        q=pair[pair.experiment.eq(a)]
        agg=q.groupby('model')[[f'delta_{m}_3y_minus_2y' for m in ['f1','pr_auc','roc_auc']]].mean().reset_index()
        agg['positive_f1_years']=agg.model.map(q.groupby('model').delta_f1_3y_minus_2y.apply(lambda x:int((x>1e-12).sum())))
        blocks += [f'### {a}: mean differences',table(agg),table(q[['model','outer_year','delta_f1_3y_minus_2y','delta_pr_auc_3y_minus_2y','delta_roc_auc_3y_minus_2y']])]
        for m,g in q.groupby('model'):
            ds=g.set_index('outer_year').delta_f1_3y_minus_2y
            peak=ds.abs().idxmax()
            blocks += [f'{m}: mean F1 delta {ds.mean():+.6f}; {int((ds>1e-12).sum())}/4 years favor 3 years, {int((ds < -1e-12).sum())}/4 favor 2 years. Largest absolute change is {peak} ({ds.loc[peak]:+.6f}); excluding that year gives mean {ds.drop(peak).mean():+.6f}.']
    blocks += ['## Tuning and threshold effects']
    p=s.pivot(index=['model','configured_window_years'],columns='approach',values='f1_mean').reset_index()
    p['tuning_gain_fixed']=p.tuned_fixed_threshold-p.baseline
    p['threshold_gain']=p.tuned_optimized_threshold-p.tuned_fixed_threshold
    blocks += [table(p), f'At mean-run level, tuning at 0.5 improves {int((p.tuning_gain_fixed>0).sum())}/16 baselines; threshold optimization improves {int((p.threshold_gain>0).sum())}/16 tuned fixed-threshold results.']
    winners=s.loc[s.groupby(['model','configured_window_years']).f1_mean.idxmax()]
    blocks += ['### Highest observed approach within each run (descriptive only)',
    'This table answers which approach scored highest within each saved experiment. It must not be used as a policy that chooses thresholds using outer-year outcomes; the rankings above evaluate the three approaches separately.',
    table(winners[['model','configured_window_years','approach','f1_mean','f1_std']])]
    for m,g in p.groupby('model'):
        blocks += [f'{m}: fixed-threshold tuning gains are '+', '.join(f'{int(r.configured_window_years)}y {r.tuning_gain_fixed:+.6f}' for r in g.itertuples())+'; threshold gains are '+', '.join(f'{int(r.configured_window_years)}y {r.threshold_gain:+.6f}' for r in g.itertuples())+'.']
    opt=f[f.experiment.eq('tuned_optimized_threshold')]
    blocks += ['### Repeated thresholds',table(opt.groupby('threshold').size().rename('fold_count').reset_index()),
    'Lower thresholds often trade precision for recall; the paired F1 changes above determine whether this helped the stated objective. A ranking-metric gain is not automatically an F1 gain.',
    '### Per-fold threshold selection and selected hyperparameters',
    'All configured runs below used 15 trials per outer fold (60 total) and the actual inner-fold counts shown. The fixed-threshold tuned model uses the same selected parameters as the optimized-threshold row. Full baseline parameter dictionaries and every metric are retained in `outer_fold_details.csv`.']
    for (model,w),g in opt.groupby(['model','configured_window_years']):
        blocks += [f'#### {model}, configured {w} years',f'Source: `{g.directory.iloc[0]}`',table(g[['outer_year','inner_folds','threshold','inner_f1','threshold_inner_f1','f1','best_parameters']])]
        params=[ast.literal_eval(x) for x in g.best_parameters]
        stability=[]
        for key in params[0]:
            vals=[d[key] for d in params]
            counts=pd.Series([repr(v) for v in vals]).value_counts()
            stability.append(dict(parameter=key,distinct_winners=len(counts),most_repeated_value=counts.index[0],repeat_count=int(counts.iloc[0])))
        blocks += [table(pd.DataFrame(stability))]
    blocks += ['## Trial-level evidence and inner-selection optimism',
    'The diagnostics inspect every saved candidate, not only winners. First-trial and last-five comparisons describe search progress, not an unbiased test of future search value. With only 15 trials and one search seed, weak progress does not prove the search space is exhausted.',
    table(ts.groupby(['model','configured_window_years'])[['unique_parameter_sets','first_trial_f1','median_inner_f1','max_inner_f1','inner_minus_outer_fixed','threshold_inner_gain','outer_threshold_gain']].mean().reset_index()),
    f'The winning trial occurs after trial 10 in {int((ts.best_trial>10).sum())}/64 outer searches. The best score in the last five exceeds the best in the first ten in {int((ts.best_last5>ts.best_first10+1e-12).sum())}/64 searches. Mean selected inner minus outer fixed F1 is {ts.inner_minus_outer_fixed.mean():+.6f}. Inner threshold gain is nonnegative in {int((ts.threshold_inner_gain>=-1e-12).sum())}/64 folds, whereas outer threshold gain is positive in {int((ts.outer_threshold_gain>1e-12).sum())}/64 folds and negative in {int((ts.outer_threshold_gain < -1e-12).sum())}/64.',
    'Inner scores reuse folds for parameter selection and threshold selection. Positive inner-to-outer gaps and inner gains that fail to transfer are compatible with selection optimism or temporal distribution shift; different time populations prevent attributing them uniquely to overfitting. The artifact does not contain per-trial outer scores (and selecting candidates with outer scores would itself introduce leakage). Numeric parameter changes are not automatically instability in predictive performance. Repeated categorical winners and distinct-value counts above are descriptive; no parameter-importance or causal claim is justified by 15 correlated search trials.',
    'HistGradientBoosting configured for 2 years illustrates ranking versus threshold behavior: tuning raises PR-AUC to 0.840126, while optimized thresholds raise F1 from 0.724051 to 0.742478 without changing PR-AUC. Conversely, Random Forest configured for 3 years improves PR-AUC through tuning (0.834692 to 0.839804), but threshold optimization lowers F1 from 0.739917 to 0.735418. Logistic Regression starts with relatively strong PR-AUC (0.836729) but baseline F1 of 0.708140; its tuned PR-AUC actually declines even as F1 improves. Ranking quality and thresholded classification therefore need separate judgments.',
    '## Historical benchmark and recommendations']
    for a in APPROACHES:
        q=ranking[ranking.approach.eq(a)]
        best=q.iloc[0]
        blocks += [f'For {a}, the highest observed mean is {best.model}, configured {int(best.configured_window_years)} years: F1 {best.f1_mean:.6f} (sample SD {best.f1_std:.6f}), delta {best.f1_mean-BENCH:+.6f} versus 0.7397. {int((q.f1_mean>BENCH).sum())}/16 rows numerically exceed that rounded reference.']
    blocks += ['The benchmark supplies only mean F1. Its PR-AUC, ROC-AUC, recall, fold results and exact unrounded F1 are not supplied as benchmark evidence here, so improvements in those metrics versus the historical benchmark cannot be established. Within these runs, those metrics can be compared to each model’s saved fixed baseline.',
    'The following table distinguishes ranking and recall changes from F1 changes due to tuning, relative to each run’s own baseline. Optimized thresholds change precision/recall but do not change probability ranking.']
    b=s[s.approach.eq('baseline')].set_index(['model','configured_window_years'])
    for a in APPROACHES[1:]:
        q=s[s.approach.eq(a)].set_index(['model','configured_window_years'])
        ds=q[[m+'_mean' for m in METRICS]]-b[[m+'_mean' for m in METRICS]]
        blocks += [f'### {a} minus own baseline',table(ds.reset_index())]
    best=ranking[ranking.approach.eq('tuned_fixed_threshold')].iloc[0]
    bestopt=ranking[ranking.approach.eq('tuned_optimized_threshold')].iloc[0]
    blocks += [f'- For further work under a predeclared threshold-0.5 policy, the observed leading tuned candidate is **{best.model}, configured {int(best.configured_window_years)} years** (F1 {best.f1_mean:.6f}). Treat it as a candidate, not a validated final winner. The optimized-threshold leader is **{bestopt.model}, configured {int(bestopt.configured_window_years)} years** (F1 {bestopt.f1_mean:.6f}). The separate rankings preserve baseline candidates too.',
    f'- Threshold optimization is not consistently beneficial: it improves mean F1 in {(p.threshold_gain>0).sum()}/16 procedures. Prefer the fixed 0.5 policy for an F1-focused default unless a specific candidate demonstrates stable transfer, or a separately defined recall/cost objective justifies a different threshold.',
    '- Neither outer sliding-window duration can be recommended from these artifacts because the actual outer training strategy is in doubt. The configured comparison can guide an inner-window hypothesis only. Do not relabel these as validated 2- and 3-year outer-window results.',
    '- Keep Random Forest as the historical reference and investigate candidates near the top of each separately declared policy, including models whose PR-AUC/ROC-AUC improve despite weaker F1. Favor stable year-by-year behavior over a one-fold peak; use the paired tables before committing to a candidate.',
    '- Pause broad additional hyperparameter searches while resolving the evaluation mismatch. Feature engineering is a reasonable next hypothesis, not an established source of improvement. Audit existing recency, purchase-count, spend, tenure and historical-gap features first; test only incremental recency/frequency/monetary interactions, trends and purchasing regularity that are available at prediction time. Preserve label maturity and fit transformations inside folds. Compare predeclared feature ablations against a frozen baseline and policy rather than expanding the model search indefinitely.',
    '- A correct temporal evaluation design must be established before any eventual new experiments. This task did not repair code or authorize/run new training. Keep the 2024 holdout untouched until the feature set, model-selection procedure and threshold policy are frozen.',
    '## Limitations and calculation notes',
    'Outer means are unweighted arithmetic means of four yearly metrics; SD is sample standard deviation (ddof=1), not a confidence interval. Four temporally related folds do not provide evidence of statistical significance. No independent significance claim is made. F1 is not pooled over customers; predictions and label prevalence are absent, so calibration, customer-level error slices and pooled metrics cannot be reconstructed.',
    'Comparing 16 configurations and inspecting three policies reuses outer validation for model selection. Even though within-run tuning uses inner folds, selecting the overall winner from these outer means creates selection bias. The observed maximum is an optimistic candidate assessment, not an unbiased final performance estimate. Baseline rows duplicated across configured windows are the same observed baseline behavior, not independent replications.',
    'Summary CSVs were checked against recomputed fold means and sample SD for every metric. Trial winners were checked against the maximum saved inner score for all 64 searches. CSV exports retain floating-point precision; Markdown rounds to six decimals. Source file SHA-256 hashes are recorded in `source_manifest.json`; rerunning `analyze.py` uses only the bounded saved batch. `fold_comparison.csv` includes all metric pairs, thresholds, training counts and selected parameters; `outer_fold_details.csv` retains all 192 original result rows with experiment identifiers. `trial_diagnostics.csv` records all 64 search summaries.',
    'Configured settings are evidence of intent, not proof that runtime code honored every setting. No execution-time code snapshot or fold date bounds were saved in these files. This prevents a definitive reconstruction of actual training date ranges without accessing raw data, which was deliberately avoided.']
    (OUT/'analysis_report.md').write_text('\n\n'.join(blocks)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
