"""One reset 180-second stochastic episode with auditable decisions and outcomes."""
import csv
import os
import time
import numpy as np
from environment import NetworkEpisode
from topology import path_edges
from policies import Policy
from config import SET,SCENARIOS,LOADS,METHODS


def run_one(n,scenario,load,seed,method,model,agent=None,save_trace=False):
    env=NetworkEpisode(n,scenario,load,seed)
    policy=Policy(method,model,agent)
    decisions=[];trace=[]
    sums={'sla':0,'collateral':0,'loss':0.,'rtt':0.,'goodput':0.}
    faulted_edges=set(env.fault_map(65)[0])|set(env.fault_map(80)[0])
    originally_affected={f.fid for f in env.flows
                        if any(e in faulted_edges for e in path_edges(env.original[f.fid]))}
    if scenario=='switch':
        # Switch incident paths are also part of the ground-truth impact set.
        originally_affected={f.fid for f in env.flows
                              if env._targets['switch'] in env.original[f.fid]}
    if scenario=='infeasible': originally_affected.add(0)
    clean_violations=0;false_alarm_ticks=0;alarm_events=0;was_alarm=False;unnecessary_changes=0
    diagnostic_hits=[];diagnostic_unconditional=[];diagnostic_jaccards=[];unknown_fault=[];unknown_normal=[]
    shap_times=[];action_times=[];install_times=[]
    initially_feasible=True
    last_harm=0;last_rollback=0
    import networkx as nx
    for sample_t in (65,80):
        actual_g=env.observed_graph(sample_t)
        if any(f.src not in actual_g or f.dst not in actual_g or not nx.has_path(actual_g,f.src,f.dst) for f in env.flows):initially_feasible=False
    first_fault_alarm=None;preemptive_alarm=None
    for t in range(SET.episode_seconds):
        obs=env.observe(t)
        proposed,_,_=policy.decide(env,obs)
        proposed['verifier_harm']=env.harmful-last_harm
        proposed['rollback']=env.rollbacks-last_rollback
        last_harm=env.harmful;last_rollback=env.rollbacks
        env.update_recovery(obs)
        truth=set(env.fault_map(t)[0])
        if t<60:clean_violations+=sum(m['viol'] for m in obs['flows'].values())
        detected=bool(proposed['detected'])
        if detected and t<60:preemptive_alarm=t
        if detected and truth and first_fault_alarm is None:first_fault_alarm=t
        if not truth:
            false_alarm_ticks+=int(detected)
            alarm_events+=int(detected and not was_alarm)
            unnecessary_changes+=int(proposed['ack']=='ack')
            unknown_normal.append(int(proposed['unknown']))
        if truth and t==65:
            diagnostic_unconditional.append(int(detected and proposed['correct_localisation']))
            if detected:diagnostic_hits.append(int(proposed['correct_localisation']))
            true_labels=set(env.active_labels(t));pred=set(filter(None,proposed['predicted_labels'].split('|')))
            diagnostic_jaccards.append(len(true_labels & pred)/max(1,len(true_labels | pred)) if detected else 0.)
            unknown_fault.append(int(proposed['unknown']))
        if proposed.get('shap_ms') is not None:shap_times.append(proposed['shap_ms'])
        was_alarm=detected
        if 60 <= t < 120:
            metrics=list(obs['flows'].values())
            sums['sla'] += sum(m['viol'] for m in metrics)
            sums['collateral'] += sum(m['viol'] for fid,m in obs['flows'].items()
                                      if fid not in originally_affected)
            sums['loss']+=sum(m['loss'] for m in metrics)
            sums['rtt']+=sum(m['rtt'] for m in metrics)
            sums['goodput']+=sum(m['goodput'] for m in metrics)
        if save_trace and 55<=t<=130:
            for fid,m in obs['flows'].items():
                trace.append((t,fid,m['rtt'],m['loss'],m['goodput'],m['viol'],
                              '-'.join(map(str,obs['routes'][fid]))))
        decisions.append(proposed)
    restored=env.first_stable is not None
    time_to_recover=env.first_stable-60 if restored else None
    service_n=60*len(env.flows)
    collision_den=60*max(1,len(env.flows)-len(originally_affected))
    feasible=scenario not in ('normal','burst','infeasible') and initially_feasible
    control=scenario in ('normal','burst')
    # These acknowledgement values refer to digital-twin route changes, NOT OpenFlow acknowledgements.
    result={'feasible':int(feasible),'restored':int(restored and feasible),
            'restoration_seconds':time_to_recover if feasible else None,
            'capped_unrecovered_seconds':min(60,time_to_recover) if restored and feasible else 60,
            'sla_burden':sums['sla']/service_n,
            'collateral_burden':sums['collateral']/collision_den if originally_affected else 0.,
            'packet_loss_percent':100*sums['loss']/service_n,
            'mean_rtt_ms':sums['rtt']/service_n,
            'mean_goodput_mbps':sums['goodput']/service_n,
            'first_alarm_second':first_fault_alarm,'preemptive_alarm_second':preemptive_alarm,
            'blocked':policy.total_blocked,'proposals':policy.total_proposed,
            'route_changes':env.route_changes,'rollbacks':env.rollbacks,
            'harmful':env.harmful,'abstentions':policy.total_abstain,
            'decision_p95_ms':float(np.percentile(policy.model_time,95)) if policy.model_time else 0.,
            'correct_infeasible_abstention':int(not feasible and not control and env.route_changes==0),
            'control':int(control),'structurally_feasible':int(initially_feasible),
            'clean_sla_burden':clean_violations/(60*len(env.flows)),
            'false_alarm_events':alarm_events,'false_alarm_ticks':false_alarm_ticks,
            'unnecessary_changes':unnecessary_changes,
            'localisation_unconditional_t65':float(np.mean(diagnostic_unconditional)) if diagnostic_unconditional else None,
            'localisation_conditional_t65':float(np.mean(diagnostic_hits)) if diagnostic_hits else None,
            'label_jaccard_t65':float(np.mean(diagnostic_jaccards)) if diagnostic_jaccards else None,
            'unknown_t65':float(np.mean(unknown_fault)) if unknown_fault else None,
            'shap_p50_ms':float(np.median(shap_times)) if shap_times else 0.,
            'shap_p95_ms':float(np.percentile(shap_times,95)) if shap_times else 0.}
    return result,decisions,trace

