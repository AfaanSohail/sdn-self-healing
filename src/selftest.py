"""Fast behavioural checks. Run before expensive experiments."""
from pathlib import Path
import numpy as np
from environment import NetworkEpisode
from features import global_features,link_features,all_link_features
from safety import check_action
from policies import Policy


def main():
 # Common random numbers match across resets; independent episode seeds differ.
 a=NetworkEpisode(12,'congestion','high',101)
 b=NetworkEpisode(12,'congestion','high',101)
 c=NetworkEpisode(12,'congestion','high',102)
 for t in range(66):
  oa=a.observe(t);ob=b.observe(t);oc=c.observe(t)
  assert oa==ob
 assert oa!=oc
 # Vectorisation preserves the per-link definition exactly.
 np.testing.assert_allclose(all_link_features(a.timeseries,a.edges),np.stack([link_features(a.timeseries,e) for e in a.edges]))
 # Window features cannot change when only future exogenous samples change.
 before=global_features(a.timeseries,a.flows).copy();a.noise[100:]=999
 np.testing.assert_array_equal(before,global_features(a.timeseries,a.flows))
 # Stale proposed route is rejected and the unshielded ablation can accept it.
 fid=a.choose_flow(oa);paths=a.paths(fid,oa)
 if paths:
  for e in list(oa['links']):oa['links'][e]['age']=4
  ok,reason=check_action(a,oa,fid,1,paths)
  assert not ok and reason=='stale_telemetry'
  assert check_action(a,oa,fid,1,paths,shield=False)[0]
 # Observable actual harm: deliberately reroute a clean flow across the active impairment.
 e=NetworkEpisode(12,'delay','low',201)
 for t in range(65):o=e.observe(t)
 import networkx as nx
 target=e._targets['main'];fid=1
 candidates=list(nx.all_simple_paths(e.graph,e.flows[fid].src,e.flows[fid].dst,cutoff=6))
 from topology import path_edges
 bad=next(p for p in candidates if target in path_edges(p))
 old=next(p for p in candidates if target not in path_edges(p))
 e.routes[fid]=old;o=e.observe(65)
 e.execute(fid,1,[bad],65);e.observe(66)
 assert e.harmful==1 and e.rollbacks==1 and e.routes[fid]==old
 # Normal controls and unresolved disconnected cases are not scored as recovered faults.
 from evaluation import run_one
 from models import FaultModels
 model=FaultModels.load(Path(__file__).resolve().parents[1]/'checkpoints/lightgbm_models.pkl')
 r,_,_=run_one(12,'infeasible','low',801,'B1_rule',model)
 assert r['restored']==0 and r['feasible']==0
 r,_,_=run_one(12,'normal','low',802,'B1_rule',model)
 assert r['control']==1 and r['feasible']==0
 # A failed endpoint switch must be counted as structurally infeasible, not crash.
 endpoint_seed=next(seed for seed in range(1,50) if NetworkEpisode(12,'switch','low',seed)._targets['switch']==6)
 r,_,_=run_one(12,'switch','low',endpoint_seed,'B1_rule',model)
 assert r['feasible']==0 and r['restored']==0
 print('PASS: seeded reset, stochastic independence, feature equivalence, causality, stale shielding, counterfactual harm/rollback, infeasibility and controls')

if __name__=='__main__':main()
