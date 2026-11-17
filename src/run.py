"""Reproduce the revised submission. Expensive runs are explicit."""
import argparse,subprocess,sys
from pathlib import Path

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--analysis-only',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--workers',type=int,default=8);args=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 def run(script,*extra):subprocess.run([sys.executable,str(root/'src'/script),*map(str,extra)],cwd=root,check=True)
 if args.full:
  run('download_public.py');run('prepare.py');run('public_benchmark.py');run('selftest.py');run('train_all.py','--steps',50000,'--workers',args.workers);run('evaluate_all.py','--blocks',30,'--workers',args.workers)
 elif not args.analysis_only:
  run('selftest.py');print('Existing results preserved. Use --analysis-only to rebuild figures or --full for the complete experiment.');sys.exit(0)
 run('analyse_final.py');run('failure_examples.py');run('audit.py');run('report.py')
