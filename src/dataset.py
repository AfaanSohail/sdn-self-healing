"""Episode-first split, fixed seed manifest, causal feature collection."""
import csv
import numpy as np
from environment import NetworkEpisode
from features import global_features,link_features,all_link_features
from config import FAULTS,DEVELOPMENT,SET

SAMPLE_TIMES = (15,35,45,55,60,61,62,63,65,68,70,75,85,95,110,118,121,126,140,160)

def collect_development(out_manifest):
    x,y,lx,ly,splits=[],[],[],[],[]
    manifest=[]
    for si,sc in enumerate(DEVELOPMENT):
        for i in range(100):
            split='train' if i<60 else ('validation' if i<80 else 'test')
            # Episode/group randomisation is fixed before constructing windows.
            seed=si*100_000 + i*19 + 1103
            load=('low','medium','high')[i%3]
            env=NetworkEpisode(12,sc,load,seed)
            manifest.append({'episode':f'{sc}_{i:03d}','seed':seed,
                             'scenario':sc,'load':load,'topology':12,'split':split})
            for t in range(SET.episode_seconds):
                obs=env.observe(t)
                if t not in SAMPLE_TIMES: continue
                faults,_=env.fault_map(t)
                x.append(global_features(env.timeseries,env.flows))
                y.append(int(bool(faults)))
                splits.append(split)
                features=all_link_features(env.timeseries,env.edges)
                for ei,e in enumerate(env.edges):
                    lx.append(features[ei])
                    target=np.zeros(len(FAULTS),dtype='uint8')
                    if e in faults and faults[e] in FAULTS:
                        target[FAULTS.index(faults[e])]=1
                    ly.append(target)
    with open(out_manifest,'w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=manifest[0].keys());w.writeheader();w.writerows(manifest)
    nlinks=len(env.edges)
    return (np.asarray(x),np.asarray(y),np.asarray(lx),np.asarray(ly),
            np.asarray(splits),nlinks)
