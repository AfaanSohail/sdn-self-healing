"""Bounded GA and discrete PSO over the shared one-flow candidate action set.
These are matched local adaptations, not reproductions of R9's GA+RF/PSO+SVM.
"""
import time
import numpy as np
from topology import path_edges
from safety import check_action

def search(name,env,obs,fid,paths):
    started=time.perf_counter()
    rng=np.random.default_rng(env.seed+obs['t']*1009+fid)
    def cost(a):
        a=int(a)
        ok,_=check_action(env,obs,fid,a,paths)
        if not ok:return 1e6
        path=env.routes[fid] if a==0 else paths[a-1]
        v=[obs['links'][e] for e in path_edges(path)]
        # Same objective for both optimizers, including no-op.
        return 2*sum(x['delay'] for x in v)+200*max(x['loss'] for x in v)+15*max(x['util'] for x in v)+.1*(a!=0)
    costs=np.array([cost(a) for a in range(len(paths)+1)])
    count=len(costs)
    population=rng.integers(count,size=20)
    # Candidate coverage is explicit; small action space can be exhaustively audited.
    population[:count]=np.arange(count)
    if name=='B3_GA':
        for _ in range(20):
            elite=population[np.argsort(costs[population])[:5]]
            offspring=rng.choice(elite,20)
            mutate=rng.random(20)<.2
            offspring[mutate]=rng.integers(count,size=mutate.sum())
            offspring[0]=elite[0];population=offspring
            if time.perf_counter()-started>.2:break
    else:
        pos=population.astype(float);vel=np.zeros(20);personal=population.copy()
        for _ in range(20):
            global_best=personal[np.argmin(costs[personal])]
            vel=.5*vel+1.5*rng.random(20)*(personal-pos)+1.5*rng.random(20)*(global_best-pos)
            pos=np.clip(pos+vel,0,count-1)
            population=np.rint(pos).astype(int)
            better=costs[population]<costs[personal];personal[better]=population[better]
            if time.perf_counter()-started>.2:break
        population=personal
    return int(population[np.argmin(costs[population])])
