"""Fail loudly on incomplete coverage, unbalanced seeds or mismatched artifacts."""
from pathlib import Path
import json,hashlib,sys,platform,importlib.metadata as md
import numpy as np
import pandas as pd
from config import SCENARIOS,LOADS,METHODS,TRAINING_SEEDS,SET

if __name__=='__main__':
 b=Path(__file__).resolve().parents[1];r=b/'results';df=pd.read_csv(r/'episodes.csv');checks={}
 checks['expected_23760_episodes']=len(df)==30*11*3*2*12
 checks['unique_method_per_block']=not df.duplicated(['block','method']).any()
 counts=df.groupby(['topology','scenario','load','method']).size()
 checks['all_792_strata_have_30_blocks']=len(counts)==2*11*3*12 and counts.eq(30).all()
 trained=df[df.training_seed.notna()]
 balance=trained.groupby(['topology','scenario','load','method','training_seed']).size()
 checks['six_blocks_per_training_seed']=balance.eq(6).all() and set(trained.training_seed)==set(TRAINING_SEEDS)
 checks['shared_environment_seed_per_block']=df.groupby('block').seed.nunique().eq(1).all()
 checks['full_180s_episode_in_source']=SET.episode_seconds==180
 manifest=pd.read_csv(r/'split_manifest.csv')
 checks['700_episode_split']=len(manifest)==700 and manifest.episode.nunique()==700
 checks['60_20_20_per_group']=all((g.split.value_counts().to_dict()=={'train':60,'validation':20,'test':20}) for _,g in manifest.groupby('scenario'))
 checks['heldout_classes_excluded']=not manifest.scenario.isin(['switch','correlated','cascading','infeasible']).any() and manifest.topology.eq(12).all()
 training=[]
 for variant in ('Full_DQN','A1_no_diagnosis','A2_no_shield','A5_no_topology'):
  for seed in TRAINING_SEEDS:
   f=b/f'checkpoints/{variant}_seed{seed}.pt';curve=pd.read_csv(str(f)+'.learning.csv')
   checks[f'{variant}_{seed}_50000_steps']=curve.step.iloc[-1]==50000 and len(curve)==5 and f.exists() and Path(str(f)+'.last.pt').exists()
   selected=int(curve[curve.selected==1].step.iloc[-1]);training.append({'variant':variant,'seed':seed,'steps':50000,'selected_step':selected,'updates':int(curve.updates.iloc[-1])})
 pivot=df.pivot(index='block',columns='method',values='sla_burden')
 checks['A6_same_service_outcomes']=np.allclose(pivot.Full_DQN,pivot.A6_no_SHAP)
 checks['B1_B2_are_distinct']=not np.allclose(pivot.B1_rule,pivot.B2_greedy)
 checks['shield_was_exercised']=df[df.method=='Full_DQN'].blocked.sum()>0
 checks['harm_verifier_was_exercised']=df.harmful.sum()>0
 checks['normal_excluded_from_primary']=df[df.scenario.isin(['normal','burst'])].feasible.eq(0).all()
 checks['infeasible_never_restored']=df[df.scenario=='infeasible'].restored.eq(0).all()
 checks['valid_sla_burdens']=df.sla_burden.between(0,1).all()
 # Preprocessing evidence with a fixed delayed-telemetry seed and explicit reset.
 from environment import NetworkEpisode
 env=NetworkEpisode(12,'congestion','high',3003);rows=[]
 for t in range(180):
  obs=env.observe(t)
  for e in {env.edges[0],env._targets['main']}:
   v=obs['links'][e];rows.append({'t':t,'edge':str(e),'bytes_counter':env.counters[e],'received':int(not v['missing']),'derived_rate_mbps':v['rate_mbps'],'missing':v['missing'],'age_s':v['age'],'reset_injected':int(t==90 and e==env.edges[0])})
 pd.DataFrame(rows).to_csv(r/'preprocessing_trace.csv',index=False)
 data=np.load(b/'data/development.npz');X=data['X']
 checks['generated_missingness_nonzero']=bool(np.max(X[:,-2])>0)
 checks['generated_staleness_above_2s']=bool(np.max(X[:,-1])>2)
 from topology import build_topology,candidate_paths
 topology_records={}
 for n in (12,20):
  graph,flows,hosts=build_topology(n)
  topology_records[str(n)]={'nodes':list(graph.nodes),'edges':[{'a':a,'b':b,**v} for a,b,v in graph.edges(data=True)],'flows':[vars(f) for f in flows],'host_attachments':hosts,'nominal_delay_note':'Actual per-episode delays are seeded uniform 1 to 3 ms; definitions and seeds reproduce them exactly.'}
 (r/'topologies.json').write_text(json.dumps(topology_records,indent=2))
 from features import GLOBAL_FEATURES,LINK_FEATURES
 schema={'global_features':GLOBAL_FEATURES,'link_features':LINK_FEATURES,'policy_state':{'0:8':'aggregate deficit, normalized RTT/loss/goodput, detector confidence, cooldown, change age and observation age','8:12':'outage, congestion, loss, delay class probabilities; zero in A1','12:27':'three candidates x hop count, residual capacity, maximum loss, normalized delay and combined utilisation/component risk','27:30':'current-route identity for each candidate'},'actions':{'0':'no-op','1:3':'candidate route indices','4':'rollback'},'scaling':'DQN state uses fixed physical-unit normalizers, not a fitted StandardScaler. SVM/IF use training-fitted StandardScaler.'}
 (r/'feature_schema.json').write_text(json.dumps(schema,indent=2))
 df[['block','replicate','topology','scenario','load','seed']].drop_duplicates().to_csv(r/'evaluation_seed_manifest.csv',index=False)
 # Include execution truth rather than treating a requested run as completed.
 out={'all_checks_passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'episodes':len(df),'blocks':df.block.nunique(),'training':training,'preprocessing':{'mean_sampled_link_missing_fraction':float(np.mean(X[:,-2])),'max_sampled_age_s':float(np.max(X[:,-1]))},'notes':['Actual class feedback: none, as confirmed by student.','External benchmark is official SpotLight platform sample only.']}
 (r/'validation.json').write_text(json.dumps(out,indent=2))
 versions={}
 for p in ('numpy','pandas','scipy','scikit-learn','lightgbm','torch','networkx','matplotlib','statsmodels','python-docx'):
  try:versions[p]=md.version(p)
  except md.PackageNotFoundError:versions[p]='unavailable in analysis runtime'
 (b/'requirements-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(f'{package}=={version}' for package,version in versions.items() if 'unavailable' not in version)+'\n')
 meta={'environment':'Stochastic Python service KPI simulation; no Mininet/OVS/OpenFlow run','python':sys.version,'platform':platform.platform(),'versions':versions,'total_evaluation_episodes':len(df),'environment_blocks_per_stratum':30,'methods':list(METHODS),'training_seeds':list(TRAINING_SEEDS),'dqn_steps_per_seed_per_variant':50000,'aggregate_dqn_environment_steps':1000000,'checkpoint_selection':'validation every 10000 steps; minimise (harm_count, SLA burden), retain earlier ties','primary_topology':12,'heldout_topology':20,'decision_interval_s':1,'fault_interval_s':[60,120],'stable_samples':5,'parallel_workers':8,'execution_order':'seeded random method order within block; multiple blocks concurrently','run_history':'Initial grid aborted on endpoint-switch connectivity classification; all partial outputs discarded. Final grid rerun in full after NodeNotFound fix; trained weights and seeds unchanged.','class_feedback':'Teacher gave no specific feedback','primary_inference':'30 environmental bundles; five frozen model seeds; hierarchical resampling by seed group','preprocessing_deviation':'Long gaps retain explicitly stale values and age; no median fallback; no probability calibration','safety_deviation':'Proposal-before-shield instrumentation; one-step counterfactual verifier with rollback, not the original two-sample harm definition','public_dataset':'Official SpotLight platform sample, chronological split, 10-row gaps; not a RANGAN replication','unexecuted':['Mininet/OpenFlow emulation','Full public-dataset benchmark','5G-NIDD','Hardware/control-plane overhead','Formal power calculation','Noise-rate robustness sweeps','Probability/unknown calibration']}
 (r/'run_manifest.json').write_text(json.dumps(meta,indent=2))
 print(json.dumps(out,indent=2))
 assert all(checks.values()),'Failed coverage/integrity checks: '+str([k for k,v in checks.items() if not v])
