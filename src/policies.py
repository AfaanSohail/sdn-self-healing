"""Matched route candidates and action budgets across recovery methods."""
import time
from config import SET
from topology import path_edges
from safety import check_action
import numpy as np

class Policy:
    def __init__(self,name,models=None,agent=None,training=False):
        self.name,self.models,self.agent=name,models,agent
        self.training=training
        self.positive_streak=0
        self.total_blocked=0
        self.total_proposed=0
        self.total_abstain=0
        self.last_state=None
        self.last_mask=None
        self.detect_at=None
        self.model_time=[]
        self.last_diag=None
        self.last_known=False

    def _rank_b1(self,env,obs,fid,paths):
        routes=env.routes[fid]
        ordered=sorted(enumerate(paths,1),key=lambda z:(len(z[1]),z[1]))
        for i,p in ordered:
            if p!=routes and check_action(env,obs,fid,i,paths)[0]: return i
        return 0

    def _rank_b2(self,env,obs,fid,paths):
        ranked=[]
        for i,p in [(0,env.routes[fid])]+list(enumerate(paths,1)):
            if i and not check_action(env,obs,fid,i,paths)[0]: continue
            seg=[obs['links'][e] for e in path_edges(p)]
            objective=sum(v['delay'] for v in seg)*2 + 200*max(v['loss'] for v in seg)
            objective+= 15*max(v['util'] for v in seg)+.1*(i!=0)
            ranked.append((objective,i))
        return min(ranked)[1] if ranked else 0

    def decide(self,env,obs,epsilon=0.):
        t0=time.perf_counter()
        changed = {'t':obs['t'],'proposed':0,'executed':0,'reason':'no_alarm',
                   'ack':'none','fault_prob':0.,'unknown':False,'flow_id':-1,
                   'predicted_labels':'','detected':False,'correct_localisation':False,
                   'shield_blocked':False,'action_ms':0.}
        if self.name == 'B0_no_recovery':
            changed['reason']='passive'
            return changed,None,None
        # Learned detector is evaluated independently of service SLA status.
        violated=any(m['viol'] for m in obs['flows'].values())
        hard_down=any(v['up']<.5 for v in obs['links'].values())
        if self.name=='B1_rule':
            probable=violated
            cls={}
            prob=1. if probable else 0.
        else:
            prob=self.models.score(env)
            probable= prob >= self.models.threshold
            cls={}
        if self.name=='A4_predictive':
            hist=list(env.timeseries)
            if len(hist)>=5:
                u=np.asarray([[x['links'][e]['util'] for e in env.edges] for x in hist[-5:]])
                # Ridge regression forecast at t+5; fixed alpha=1, validated threshold=.95.
                slope=((np.arange(5)-2)[:,None]*(u-u.mean(0))).sum(0)/11
                probable=probable or bool(np.max(u[-1]+5*slope)>.95)
        self.positive_streak=self.positive_streak+1 if probable else 0
        if (self.positive_streak<2 and not (hard_down and violated)):
            changed.update(fault_prob=prob,reason='await_confirmation')
            return changed,None,None
        if self.detect_at is None:
            self.detect_at=obs['t']
        changed.update(fault_prob=prob,detected=True)
        if self.name not in ('B1_rule',):
            cls,link_list,unknown=self.models.diagnose(env)
            changed['unknown']=unknown
            changed['predicted_labels']='|'.join(k for k,v in cls.items()
                                    if v>=self.models.fault_threshold[k])
            truth=set(env.fault_map(obs['t'])[0]) # evaluator field only, never used for actions
            changed['correct_localisation'] = bool(link_list and link_list[0][0] in truth)
            if self.name=='Oracle':
                cls={k:float(k in env.active_labels(obs['t'])) for k in ('outage','congestion','loss','delay')}
            if self.name!='A6_no_SHAP' and (not self.training):
                from features import global_features,GLOBAL_FEATURES
                import json
                ts=time.perf_counter()
                contrib=self.models.detector.booster_.predict(global_features(env.timeseries,env.flows).reshape(1,-1),pred_contrib=True,num_threads=1)[0]
                changed['shap_ms']=1000*(time.perf_counter()-ts)
                changed['shap_top']=json.dumps([{"feature":GLOBAL_FEATURES[i],"contribution":float(contrib[i])} for i in np.argsort(-abs(contrib[:-1]))[:5]])
        else: link_list=[]
        env.diag_scores={} if self.name=='A1_no_diagnosis' else {e:max(v.values()) for e,v in link_list}
        if self.name=='Oracle': env.diag_scores={e:1. for e in env.fault_map(obs['t'])[0]}
        fid=env.choose_flow(obs)
        paths=env.paths(fid,obs)
        changed['flow_id']=fid
        state=env.get_state(obs,prob,cls,paths,fid,
                            ignore_diag=self.name=='A1_no_diagnosis')
        if self.name=='A5_no_topology':
            for i in range(3): state[12+5*i:17+5*i]=0.
        shield=self.name!='A2_no_shield'
        masks=[check_action(env,obs,fid,a,paths,shield=shield)[0] for a in range(5)]
        if self.name=='B1_rule': action=self._rank_b1(env,obs,fid,paths)
        elif self.name=='B2_greedy': action=self._rank_b2(env,obs,fid,paths)
        elif self.name in ('B3_GA','B4_PSO'):
            from optimizers import search
            action=search(self.name,env,obs,fid,paths)
        else:
            # Proposals are generated before shielding, so interventions are observable.
            proposal_mask=[True]+[i<len(paths) for i in range(3)]+[fid in env.previous]
            action=self.agent.choose(state,proposal_mask,eps=epsilon)
        changed['proposed']=action
        if action != 0: self.total_proposed+=1
        permitted,reason=check_action(env,obs,fid,action,paths,shield=shield)
        if not permitted:
            self.total_blocked+=1
            changed.update(shield_blocked=True,reason=reason)
            action=0
        if action==0:
            self.total_abstain+=1
            changed['reason']=reason if changed['shield_blocked'] else 'no_change'
        else:
            result=env.execute(fid,action,paths,obs['t'])
            changed.update(executed=action,ack=result,reason=reason)
        changed['action_ms']=1000*(time.perf_counter()-t0)
        self.model_time.append(changed['action_ms'])
        return changed,state,masks
