"""Episode/block inference; no sample-level pseudoreplication."""
from pathlib import Path
import json,pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_plot_style import apply
apply()
from scipy.stats import friedmanchisquare,wilcoxon,rankdata
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve,precision_recall_curve,ConfusionMatrixDisplay

B=Path(__file__).resolve().parents[1];R=B/'results';F=R/'figures';F.mkdir(exist_ok=True)
LABELS={'B0_no_recovery':'B0 None','B1_rule':'B1 Shortest','B2_greedy':'B2 Greedy','B3_GA':'B3 GA','B4_PSO':'B4 PSO','Full_DQN':'Full DQN','A1_no_diagnosis':'A1 No diagnosis','A2_no_shield':'A2 No shield','A4_predictive':'A4 Predictive','A5_no_topology':'A5 No topology','A6_no_SHAP':'A6 No SHAP','Oracle':'Oracle'}
ORDER=list(LABELS)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})

def save(name):
 plt.tight_layout();plt.savefig(F/f'{name}.png',dpi=180,bbox_inches='tight');plt.close()

def hierarchical_ci(values,seed=123):
 # 30 rows: 5 trained seeds, each with 6 independent environment schedules.
 values=np.asarray(values);rng=np.random.default_rng(seed);out=[]
 groups=[values[np.arange(len(values))%5==s] for s in range(5)]
 for _ in range(10000):
  chosen=rng.integers(0,5,5);out.append(np.mean([rng.choice(groups[s],len(groups[s]),replace=True).mean() for s in chosen]))
 return np.quantile(out,[.025,.975]).tolist()

def curve_plot(filename,prefix):
 with open(R/filename,'rb') as f:curves=pickle.load(f)
 fig,ax=plt.subplots(1,2,figsize=(9,3))
 for name,(y,s,p) in curves.items():
  fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s)
  ax[0].plot(fpr,tpr,label=name);ax[1].plot(re,pr,label=name)
 ax[0].plot([0,1],[0,1],':',c='grey');ax[0].set(xlabel='False positive rate',ylabel='True positive rate',title=prefix+' ROC')
 ax[1].set(xlabel='Recall',ylabel='Precision',title=prefix+' Precision recall');ax[1].legend(fontsize=7)
 save(prefix.lower().replace(' ','_')+'_curves')
 fig,ax=plt.subplots(figsize=(3.6,3));y,s,p=curves['LightGBM'];ConfusionMatrixDisplay.from_predictions(y,p,ax=ax,colorbar=False,display_labels=['Normal','Fault']);ax.set_title(prefix+' LightGBM');save(prefix.lower().replace(' ','_')+'_confusion')

if __name__=='__main__':
 df=pd.read_csv(R/'episodes.csv')
 df['model_seed_index']=df.replicate%5
 primary=df[(df.feasible==1)&(df.scenario!='switch')]
 metrics=['sla_burden','capped_unrecovered_seconds','restored','collateral_burden','harmful','blocked','route_changes','decision_p95_ms','shap_p50_ms','false_alarm_events','unnecessary_changes','localisation_unconditional_t65','localisation_conditional_t65','label_jaccard_t65','unknown_t65']
 df.groupby(['topology','scenario','load','method'],sort=False)[metrics].agg(['mean','std','count']).to_csv(R/'stratified_metrics.csv')
 primary.groupby(['topology','method','model_seed_index'])[metrics].mean().to_csv(R/'per_training_seed.csv')
 summary=[]
 for (n,m),g in primary.groupby(['topology','method']):
  row={'topology':int(n),'method':m,'episodes':len(g)}
  for metric in metrics:
   value=g[metric].mean();row[metric]=None if pd.isna(value) else float(value)
  for metric in ('sla_burden','capped_unrecovered_seconds'):
   vals=g.groupby('replicate')[metric].mean().reindex(range(30)).to_numpy()
   row[metric+'_CI95']=hierarchical_ci(vals)
  rec=g[g.restored==1].restoration_seconds
  row['conditional_restoration_median']=float(rec.median()) if len(rec) else None
  row['conditional_restoration_p95']=float(rec.quantile(.95)) if len(rec) else None
  summary.append(row)
 with open(R/'summary.json','w') as f:json.dump(summary,f,indent=2)
 tests=[];omnibus={}
 for metric in ('sla_burden','capped_unrecovered_seconds'):
  g=primary[primary.topology==12].groupby(['replicate','method'])[metric].mean().unstack()
  methods=['B1_rule','B2_greedy','B3_GA','B4_PSO','Full_DQN']
  result=friedmanchisquare(*[g[m] for m in methods])
  omnibus[metric]={'statistic':float(result.statistic),'p':float(result.pvalue),'matched_schedule_blocks':len(g),'methods':methods,'excludes_B0':True}
  for baseline in methods[:-1]:
   d=(g.Full_DQN-g[baseline]).to_numpy();nz=d[d!=0]
   p=float(wilcoxon(d).pvalue) if len(nz) else 1.
   ranks=rankdata(abs(nz));effect=float(np.sum(ranks*np.sign(nz))/ranks.sum()) if len(nz) else 0.
   tests.append({'metric':metric,'baseline':baseline,'mean_difference':float(d.mean()),'median_difference':float(np.median(d)),'CI95_hierarchical':hierarchical_ci(d),'raw_p':p,'rank_biserial_positive_means_full_worse':effect,'omnibus_rejects':bool(result.pvalue<.05)})
 adjusted=multipletests([t['raw_p'] for t in tests],method='holm')[1]
 for t,p in zip(tests,adjusted):t['Holm_p']=float(p)
 with open(R/'paired_statistics.json','w') as f:json.dump({'unit':'30 independent environmental schedule bundles, averaged equally over 7 fault x 3 load strata on N12','interpretation':'Tests condition on the five frozen trained models. Hierarchical intervals resample five model seed groups and six schedule bundles per group. Only five training replicates; p values do not establish population-wide algorithm superiority.','omnibus':omnibus,'contrasts':tests},f,indent=2)
 secondary=[]
 for metric in ('sla_burden','capped_unrecovered_seconds','harmful'):
  wide=primary[primary.topology==12].groupby(['replicate','method'])[metric].mean().unstack()
  for variant in ('A1_no_diagnosis','A2_no_shield','A4_predictive','A5_no_topology','A6_no_SHAP','Oracle'):
   delta=(wide[variant]-wide.Full_DQN).to_numpy()
   raw=float(wilcoxon(delta).pvalue) if np.any(delta!=0) else 1.
   secondary.append({'variant':variant,'metric':metric,'mean_variant_minus_full':float(delta.mean()),'CI95_hierarchical':hierarchical_ci(delta),'raw_p':raw})
 correction=multipletests([x['raw_p'] for x in secondary],method='holm')[1]
 for item,q in zip(secondary,correction):item['Holm_p']=float(q)
 (R/'ablation_statistics.json').write_text(json.dumps({'scope':'Exploratory secondary family of 18 variant/endpoint contrasts, conditional on five frozen seeds','contrasts':secondary},indent=2))
 # Exploratory sensitivity: retain complete blocks whose passive pre-injection
 # service was healthy. This also removes strong congestion precursor episodes.
 healthy_blocks=set(df[(df.method=='B0_no_recovery')&df.clean_sla_burden.eq(0)].block)
 sensitivity=primary[primary.block.isin(healthy_blocks)].groupby(['topology','method'])[['sla_burden','capped_unrecovered_seconds','restored']].agg(['mean','count'])
 sensitivity.to_csv(R/'healthy_preinjection_sensitivity.csv')
 df[df.method=='B0_no_recovery'].groupby(['topology','scenario','load']).clean_sla_burden.agg(['mean','max']).to_csv(R/'passive_preinjection_burden.csv')
 # B0 is retained in tables; use active-policy scales to expose smaller differences.
 from recovery_figures import render_recovery_figures
 render_recovery_figures()
 core=ORDER[:6]
 fig,ax=plt.subplots(figsize=(7,3.5))
 for m in core[1:]:
  a=primary[(primary.topology==12)&(primary.method==m)]
  x=np.linspace(0,60,121);ax.plot(x,[np.mean(a.restored.eq(1)&a.restoration_seconds.le(t)) for t in x],label=LABELS[m])
 ax.set(xlabel='Seconds after injection',ylabel='Fraction restored before clearance',ylim=(0,1.02),title='Recovery CDF including unrecovered episodes');ax.legend(ncol=2,fontsize=8);save('restoration_cdf')
 fig,ax=plt.subplots(figsize=(8,3.8));abl=['Full_DQN','A1_no_diagnosis','A2_no_shield','B2_greedy','A4_predictive','A5_no_topology','A6_no_SHAP','Oracle']
 rows={r['method']:r for r in summary if r['topology']==12};means=np.array([100*rows[m]['sla_burden'] for m in abl]);ci=np.array([rows[m]['sla_burden_CI95'] for m in abl]).T*100
 ax.bar(range(8),means,color=plt.cm.Set2(np.arange(8)));ax.errorbar(range(8),means,yerr=np.maximum(0,np.vstack([means-ci[0],ci[1]-means])),fmt='none',c='black',capsize=3);ax.set(xticks=range(8),xticklabels=[LABELS[m] for m in abl],ylabel='SLA burden (%)',title='Ablations: A3 is B2 greedy');ax.tick_params(axis='x',rotation=30);save('ablations')
 fig,ax=plt.subplots(figsize=(7,3.6))
 for m in ['Full_DQN','A2_no_shield','B1_rule','B2_greedy','A4_predictive']:
  g=primary[(primary.topology==12)&(primary.method==m)];x=g.capped_unrecovered_seconds.mean();y=100*g.harmful.sum()/max(1,g.route_changes.sum()-g.rollbacks.sum());ax.scatter(x,y,s=55,label=LABELS[m])
 ax.set(xlabel='Mean capped time unrecovered (s)',ylabel='Harm events / 100 route installs',title='Safety and restoration speed');ax.legend(fontsize=8);ax.margins(.15);save('safety_speed')
 fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True)
 for ax,v in zip(axes.flat,['Full_DQN','A1_no_diagnosis','A2_no_shield','A5_no_topology']):
  for seed in (11,22,33,44,55):
   a=pd.read_csv(B/f'checkpoints/{v}_seed{seed}.pt.learning.csv');ax.plot(a.step,a.validation_sla*100,marker='.',label=str(seed))
  ax.set(title=LABELS[v],ylabel='Validation SLA burden (%)',xlabel='Environment steps');ax.legend(fontsize=6,ncol=5)
 save('learning_curves')
 curve_plot('detector_curves.pkl','Synthetic');curve_plot('spotlight_curves.pkl','SpotLight sample')
 # Controls, stage transitions, audit and A6 identity check.
 controls=df[df.control==1].groupby(['topology','scenario','method'])[['false_alarm_events','unnecessary_changes','sla_burden','harmful']].mean();controls['false_alarm_events_per_hour']=20*controls.false_alarm_events;controls['unnecessary_changes_per_hour']=20*controls.unnecessary_changes;controls.to_csv(R/'negative_controls.csv')
 stages=[]
 for (n,m),g in df[df.feasible==1].groupby(['topology','method']):
  detected=g.first_alarm_second.notna();diagnosed=detected & g.localisation_unconditional_t65.eq(1);executed=diagnosed & g.route_changes.gt(0);safe=executed & g.harmful.eq(0);restored=safe & g.restored.eq(1)
  previous=len(g)
  for stage,mask in [('detected',detected),('localised_t65',diagnosed),('executed',executed),('safe',safe),('restored',restored)]:
   count=int(mask.sum());stages.append({'topology':n,'method':m,'stage':stage,'count':count,'all_faults':len(g),'fraction_all':count/len(g),'fraction_previous':count/previous if previous else None});previous=count
 pd.DataFrame(stages).to_csv(R/'stage_transition.csv',index=False)
 pivot=df.pivot(index='block',columns='method',values='sla_burden')
 audit={'episodes':len(df),'blocks':int(df.block.nunique()),'methods':df.method.nunique(),'complete_paired_blocks':bool(df.groupby('block').method.nunique().eq(12).all()),'B1_B2_nonidentical_SLA_fraction':float((pivot.B1_rule!=pivot.B2_greedy).mean()),'A6_identical_SLA':bool(np.allclose(pivot.Full_DQN,pivot.A6_no_SHAP)),'independent_baseline_result_counts':df[df.method=='B1_rule'].groupby(['topology','scenario','load']).sla_burden.nunique().to_dict()}
 audit['independent_baseline_result_counts']={str(k):int(v) for k,v in audit['independent_baseline_result_counts'].items()}
 audit['shield_blocks_full']=int(df[df.method=='Full_DQN'].blocked.sum());audit['harm_full']=int(df[df.method=='Full_DQN'].harmful.sum());audit['harm_A2']=int(df[df.method=='A2_no_shield'].harmful.sum())
 with open(R/'evaluation_audit.json','w') as f:json.dump(audit,f,indent=2)
 # Failure exemplars and first observed failure stage.
 failures=df[(df.feasible==1)&(df.restored==0)].copy()
 failures['first_observed_stage']=np.select([failures.first_alarm_second.isna(),failures.localisation_unconditional_t65.eq(0),failures.blocked.gt(0),failures.harmful.gt(0)],['missed_detection','localisation_at_t65','shield_or_policy','harm_or_verification'],default='policy_or_capacity')
 failures.to_csv(R/'failed_episodes.csv',index=False)
 for scenario in ('congestion','cascading','switch'):
  try:
   trace=pd.read_csv(R/'telemetry_sample.csv');bid=trace.block[trace.block.str.contains('N12_'+scenario+'_high')].iloc[0];g=trace[trace.block==bid]
   fig,ax=plt.subplots(figsize=(8,3))
   for m in ['B1_rule','B2_greedy','Full_DQN','A2_no_shield']:
    a=g[g.method==m].groupby('t').violated.mean();ax.plot(a.index,100*a.values,label=LABELS[m],alpha=.8,linestyle={'B1_rule':'-','B2_greedy':'--','Full_DQN':':','A2_no_shield':'-.'}[m])
   ax.axvspan(60,120,alpha=.1,color='red');ax.set(xlabel='Episode second',ylabel='Violating flows (%)',title=f'{scenario}: one matched high-load trace');ax.legend(ncol=2,fontsize=8);save('timeline_'+scenario)
  except IndexError:pass
 print(json.dumps(audit,indent=2))
