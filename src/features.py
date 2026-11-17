"""Causal 10-s windows. Every feature is observed at or before time t."""
import numpy as np
from topology import edge

SIGNALS = ('util','loss','delay','rate_mbps')
SUMMARY = ('mean','max','std','trend')
LINK_FEATURES = [f'{s}_{q}' for s in SIGNALS for q in SUMMARY] + [
    'up_now','up_min','neighbour_util','neighbour_loss','age','history_len','missing']
GLOBAL_FEATURES = [f'{s}_{q}' for s in ('rtt','loss','goodput_ratio','violation')
                   for q in SUMMARY] + [
    'max_link_util','max_link_loss','max_link_delay','min_link_up',
    'mean_link_util','mean_link_loss','violating_flows','history_len','missing_fraction','max_age']

def summarize(values):
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return [float(a[-1])]*2 + [0.,0.]
    x = np.arange(len(a),dtype=float)
    slope = float(np.dot(x-x.mean(),a-a.mean())/sum((x-x.mean())**2))
    return [float(a.mean()),float(a.max()),float(a.std()),slope]

def global_features(history, flows):
    hist = list(history)
    by_t = []
    for s in hist:
        m = s['flows']
        by_t.append({
            'rtt': max(v['rtt'] for v in m.values()),
            'loss': max(v['loss'] for v in m.values()),
            'goodput_ratio': min(v['goodput']/flows[i].demand for i,v in m.items()),
            'violation': sum(v['viol'] for v in m.values())/len(flows)})
    result = [z for key in ('rtt','loss','goodput_ratio','violation')
              for z in summarize([s[key] for s in by_t])]
    last = hist[-1]
    links = list(last['links'].values())
    result += [max(v['util'] for v in links),max(v['loss'] for v in links),
               max(v['delay'] for v in links),min(v['up'] for v in links),
               float(np.mean([v['util'] for v in links])),
               float(np.mean([v['loss'] for v in links])),
               sum(m['viol'] for m in last['flows'].values()),len(hist)/10.,float(np.mean([v.get('missing',0) for v in links])),max(v['age'] for v in links)]
    return np.asarray(result,dtype='float32')

def link_features(history, e):
    hist = list(history)
    vals = [o['links'][e] for o in hist]
    result = [z for sig in SIGNALS for z in summarize([v[sig] for v in vals])]
    linked = [v for a,v in hist[-1]['links'].items() if a != e and
              (e[0] in a or e[1] in a)]
    result += [vals[-1]['up'], min(v['up'] for v in vals),
               float(np.mean([v['util'] for v in linked])) if linked else 0.,
               float(np.mean([v['loss'] for v in linked])) if linked else 0.,
               vals[-1]['age'],len(hist)/10.,vals[-1].get('missing',0)]
    return np.asarray(result,dtype='float32')

def all_link_features(history,edges):
    """Vectorised implementation of link_features; same causal feature order."""
    hist=list(history)
    a=np.asarray([[[h['links'][e][s] for s in SIGNALS] for e in edges] for h in hist])
    x=np.arange(len(hist),dtype=float);x-=x.mean()
    slope=np.einsum('t,tes->es',x,a)/max(1.,np.dot(x,x))
    stats=np.stack([a.mean(0),a.max(0),a.std(0),slope],axis=-1).reshape(len(edges),-1)
    last=hist[-1]['links']
    adjacency=np.asarray([[float(i!=j and (e[0] in z or e[1] in z)) for j,z in enumerate(edges)] for i,e in enumerate(edges)])
    den=np.maximum(1,adjacency.sum(1))
    nu=adjacency@np.array([last[e]['util'] for e in edges])/den
    nl=adjacency@np.array([last[e]['loss'] for e in edges])/den
    tail=np.asarray([[last[e]['up'],min(h['links'][e]['up'] for h in hist),nu[i],nl[i],last[e]['age'],len(hist)/10.,last[e].get('missing',0)] for i,e in enumerate(edges)])
    return np.column_stack([stats,tail]).astype('float32')
