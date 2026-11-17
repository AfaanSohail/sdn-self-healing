"""Online one-second DQN transitions, validation-only checkpoint selection."""
import csv
import random
import numpy as np
from environment import NetworkEpisode
from config import TRAINING_SEEDS
from agents import DQN
from safety import check_action

# Training and validation schedules are disjoint from supervised and final test seeds.
TRAIN_SCENARIOS=('normal','burst','outage','congestion','loss','delay','concurrent')

def encode(env,obs,model,variant):
    fid=env.choose_flow(obs);paths=env.paths(fid,obs)
    prob=model.score(env)
    cls,links,_=model.diagnose(env)
    env.diag_scores={} if variant=='A1_no_diagnosis' else {e:max(v.values()) for e,v in links}
    state=env.get_state(obs,prob,cls,paths,fid,ignore_diag=variant=='A1_no_diagnosis')
    if variant=='A5_no_topology':
        state[12:27]=0.
    return state,fid,paths

def validation(agent,models,variant):
    from evaluation import run_one
    rows=[]
    for i,sc in enumerate(('normal','burst','outage','congestion','loss','delay','concurrent')):
        r,_,_=run_one(12,sc,('low','medium','high')[i%3],8_000_001+i,variant,models,agent)
        rows.append(r)
    return float(np.mean([r['sla_burden'] for r in rows])),sum(r['harmful'] for r in rows)

def train_agent(seed,models,checkpoint,ablation=None,steps=50000):
    variant=ablation or 'Full_DQN';agent=DQN(seed)
    rng=random.Random(seed);tick=0;episode=0;best=None;curve=[]
    while tick<steps:
        sc=rng.choice(TRAIN_SCENARIOS);load=rng.choice(('low','medium','high'))
        env=NetworkEpisode(12,sc,load,1_000_000+seed*10000+episode)
        # Full 180-second episodes, including normal controls.
        obs=env.observe(0);state,fid,paths=encode(env,obs,models,variant)
        reward_sum=0.;updates=0;positive_streak=0
        for t in range(179):
            epsilon=max(.05,1-.95*tick/max(1,.7*steps))
            proposal_mask=[True]+[i<len(paths) for i in range(3)]+[fid in env.previous]
            action=agent.choose(state,proposal_mask,eps=epsilon)
            # The same detector trigger is enforced in training and evaluation.
            positive_streak=positive_streak+1 if state[4]>=models.threshold else 0
            hard=any(v['up']<.5 for v in obs['links'].values()) and any(v['viol'] for v in obs['flows'].values())
            if positive_streak<2 and not hard:action=0
            permitted,_=check_action(env,obs,fid,action,paths,shield=variant!='A2_no_shield')
            executed=action if permitted else 0
            before_harm=env.harmful;before_changes=env.route_changes
            env.execute(fid,executed,paths,t)
            obs2=env.observe(t+1)
            first=env.first_stable;env.update_recovery(obs2)
            reward=env.current_reward(obs2,changed=env.route_changes>before_changes,
                 harmful=env.harmful>before_harm,recovered_bonus=first is None and env.first_stable is not None)
            ns,nfid,npaths=encode(env,obs2,models,variant)
            # Executed consequences train the proposed action including rejection/no-op.
            nextmask=[True]+[i<len(npaths) for i in range(3)]+[nfid in env.previous]
            done=t==178 or tick+1>=steps
            agent.remember(state,action,reward,ns,nextmask,done)
            if tick%4==0:agent.update()
            reward_sum+=reward;tick+=1
            state,fid,paths,obs=ns,nfid,npaths,obs2
            if tick%10000==0 or tick==steps:
                score,harm=validation(agent,models,variant)
                # Safety-first lexicographic validation objective, then SLA burden.
                key=(harm,score)
                chosen=best is None or key<best
                if chosen:best=key;agent.save(checkpoint)
                curve.append(dict(step=tick,validation_sla=score,validation_harm=harm,
                                  selected=int(chosen),training_reward_per_tick=reward_sum/(t+1),
                                  updates=agent.learn_steps))
                with open(str(checkpoint)+'.learning.csv','w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=curve[0].keys());w.writeheader();w.writerows(curve)
                print(variant,seed,tick,'val',score,'harm',harm,flush=True)
            if tick>=steps:break
        episode+=1
    agent.save(str(checkpoint)+'.last.pt')
    return DQN.load(checkpoint)
