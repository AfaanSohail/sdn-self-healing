import argparse,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def worker(task):
 from models import FaultModels
 from training import train_agent
 b=Path(__file__).resolve().parents[1];variant,seed,steps=task
 model=FaultModels.load(b/'checkpoints'/('no_topology_models.pkl' if variant=='A5_no_topology' else 'lightgbm_models.pkl'))
 train_agent(seed,model,b/f'checkpoints/{variant}_seed{seed}.pt',variant,steps)
 return variant,seed

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=50000);ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
 from config import TRAINING_SEEDS
 tasks=[(v,s,args.steps) for v in ('Full_DQN','A1_no_diagnosis','A2_no_shield','A5_no_topology') for s in TRAINING_SEEDS]
 start=time.time()
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for r in pool.map(worker,tasks):print('DONE',r,'elapsed',time.time()-start,flush=True)
