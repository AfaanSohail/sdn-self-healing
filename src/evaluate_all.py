"""Final fixed-test schedules: 30 blocks x 11 scenarios x 3 loads x 2 topologies.
Five training seeds rotate evenly: six independent environment blocks per seed/stratum.
"""
import argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from config import SCENARIOS,LOADS,METHODS,TRAINING_SEEDS

_worker_models=None
_worker_agents=None

def init_worker():
 global _worker_models,_worker_agents
 from models import FaultModels
 from agents import DQN
 b=Path(__file__).resolve().parents[1]
 _worker_models={v:FaultModels.load(b/'checkpoints'/f) for v,f in [('full','lightgbm_models.pkl'),('topo','no_topology_models.pkl')]}
 _worker_agents={(v,s):DQN.load(b/f'checkpoints/{v}_seed{s}.pt') for v in ('Full_DQN','A1_no_diagnosis','A2_no_shield','A5_no_topology') for s in TRAINING_SEEDS}

def worker(task):
 from evaluation import run_one
 n,scenario,load,block=task
 seed=2_000_003+100003*block+3013*SCENARIOS.index(scenario)+73*LOADS.index(load)+11*n
 bid=f'N{n}_{scenario}_{load}_{block:02d}';ts=TRAINING_SEEDS[block%5]
 rows=[];decisions=[];traces=[]
 import random
 method_order=list(METHODS);random.Random(seed).shuffle(method_order)
 for method in method_order:
  variant='Full_DQN' if method in ('A4_predictive','A6_no_SHAP','Oracle') else method
  model=_worker_models['topo' if method=='A5_no_topology' else 'full']
  ag=_worker_agents.get((variant,ts))
  r,logs,trace=run_one(n,scenario,load,seed,method,model,ag,save_trace=(block==0 and method in ('B1_rule','B2_greedy','Full_DQN','A2_no_shield')))
  r.update(block=bid,replicate=block,topology=n,scenario=scenario,load=load,seed=seed,method=method,training_seed=ts if ag else '')
  rows.append(r)
  # All alarms/actions plus first and last interval: compact but auditable.
  for d in logs:
   if d['proposed'] or d['detected'] or d['t'] in (0,59,60,119,179):
    decisions.append(dict(block=bid,method=method,**d))
  for t,fid,rtt,loss,gput,viol,path in trace:
   traces.append(dict(block=bid,method=method,t=t,flow_id=fid,rtt_ms=rtt,loss_frac=loss,goodput_mbps=gput,violated=viol,path=path))
 return rows,decisions,traces

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--blocks',type=int,default=30);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
 b=Path(__file__).resolve().parents[1];start=time.time()
 tasks=[(n,s,l,r) for n in (12,20) for s in SCENARIOS for l in LOADS for r in range(args.blocks)]
 with open(b/'results/episodes.csv','w',newline='') as ef,open(b/'results/decisions.csv','w',newline='') as df,open(b/'results/telemetry_sample.csv','w',newline='') as tf:
  ew=dw=tw=None
  with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
   for i,(rows,ds,trace) in enumerate(pool.map(worker,tasks,chunksize=1)):
    if ew is None:
     ew=csv.DictWriter(ef,fieldnames=rows[0].keys());ew.writeheader()
     keys=sorted(set().union(*(d.keys() for d in ds)))
     # SHAP keys may be absent in early tasks.
     keys=sorted(set(keys)|{'shap_ms','shap_top'})
     dw=csv.DictWriter(df,fieldnames=keys,extrasaction='ignore');dw.writeheader()
     tw=csv.DictWriter(tf,fieldnames=['block','method','t','flow_id','rtt_ms','loss_frac','goodput_mbps','violated','path']);tw.writeheader()
    ew.writerows(rows);dw.writerows(ds);tw.writerows(trace)
    if (i+1)%30==0:
     ef.flush();df.flush();tf.flush()
     print('BLOCKS',i+1,'/',len(tasks),'episodes',(i+1)*len(METHODS),'elapsed_s',round(time.time()-start),flush=True)
 print('FINAL_EVALUATION_DONE',time.time()-start,flush=True)
