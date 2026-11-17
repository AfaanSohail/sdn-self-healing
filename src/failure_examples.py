"""Replay up to five matched examples per observed failure/stress category.
Uses frozen checkpoints; verifies service outcomes against the final evaluation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_plot_style import apply
apply()
from agents import DQN
from models import FaultModels
from evaluation import run_one

if __name__=='__main__':
 b=Path(__file__).resolve().parents[1];r=b/'results';out=r/'failure_examples';out.mkdir(exist_ok=True)
 df=pd.read_csv(r/'episodes.csv');fail=pd.read_csv(r/'failed_episodes.csv');full=df[df.method=='Full_DQN']
 selected=[]
 for label,g in fail[fail.method=='Full_DQN'].groupby('first_observed_stage'):
  selected.extend((str(label),row) for _,row in g.head(5).iterrows())
 for label,g in [('cascading',full[full.scenario=='cascading']),('unseen_switch',full[full.scenario=='switch']),('benign_false_trigger',df[(df.scenario=='burst')&(df.method=='A4_predictive')&(df.false_alarm_events>0)])]:
  selected.extend((label,row) for _,row in g.head(5).iterrows())
 models=FaultModels.load(b/'checkpoints/lightgbm_models.pkl');agents={};manifest=[];plotted=False
 for label,row in selected:
  method=row.method;variant='Full_DQN' if method in ('A4_predictive','A6_no_SHAP','Oracle') else method
  seed=int(row.training_seed);key=(variant,seed)
  if key not in agents:agents[key]=DQN.load(b/f'checkpoints/{variant}_seed{seed}.pt')
  result,logs,trace=run_one(int(row.topology),row.scenario,row.load,int(row.seed),method,models,agents[key],save_trace=True)
  for metric in ['sla_burden','restored','capped_unrecovered_seconds','harmful','blocked','route_changes']:
   assert np.isclose(result[metric],row[metric]),(row.block,method,metric,result[metric],row[metric])
  stem=f'{label}_{row.block}_{method}'
  pd.DataFrame(logs).to_csv(out/(stem+'_decisions.csv'),index=False)
  tt=pd.DataFrame(trace,columns=['t','flow_id','rtt_ms','loss_frac','goodput_mbps','violated','path']);tt.to_csv(out/(stem+'_telemetry.csv'),index=False)
  manifest.append({'category':label,'block':row.block,'method':method,'seed':int(row.seed),'training_seed':seed,'service_replay_matches':True,'file_prefix':stem})
  if not plotted and row.restored==0 and row.feasible==1 and method=='Full_DQN':
   fig,axes=plt.subplots(3,1,figsize=(8,5),sharex=True,gridspec_kw={'height_ratios':[2,2,1]})
   for fid,g in tt.groupby('flow_id'):
    axes[0].plot(g.t,g.rtt_ms,label=f'Flow {fid}');axes[1].plot(g.t,100*g.loss_frac)
   axes[0].axhline(20,ls=':',c='black');axes[1].axhline(1,ls=':',c='black')
   dd=pd.DataFrame(logs);ex=dd[dd.ack=='ack'];blocked=dd[dd.shield_blocked==True]
   axes[2].scatter(ex.t,np.ones(len(ex)),marker='o',s=15,label='Installed')
   axes[2].scatter(blocked.t,np.zeros(len(blocked)),marker='x',s=18,label='Blocked')
   axes[2].set(yticks=[0,1],yticklabels=['Blocked','Installed'],xlabel='Episode second',xlim=(55,130))
   axes[0].set(ylabel='RTT (ms)',title='Unrecovered example: '+row.block);axes[1].set(ylabel='Loss (%)');axes[0].legend(ncol=3,fontsize=8)
   for ax in axes:ax.axvspan(60,120,color='red',alpha=.08)
   plt.tight_layout();plt.savefig(r/'figures/failure_detail.png',dpi=180,bbox_inches='tight');plt.close();plotted=True
 (r/'failure_examples_manifest.json').write_text(json.dumps(manifest,indent=2))
 print('Replayed and checked',len(manifest),'representative cases',flush=True)
