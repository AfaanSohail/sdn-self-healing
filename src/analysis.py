"""Episode-disjoint supervised detection and localisation metrics."""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare,wilcoxon
from statsmodels.stats.multitest import multipletests
from config import FAULTS

BASELINES=['B1_rule','B2_greedy']


def save_detection(base,models,data):
    from sklearn.metrics import precision_recall_fscore_support,accuracy_score,f1_score
    from models import detection_baselines
    X,y,LX,LY,splits,nlinks=data
    ti=(splits=='train'); vi=(splits=='validation'); te=(splits=='test')
    trainlinks=np.repeat(ti,nlinks);validlinks=np.repeat(vi,nlinks);testlinks=np.repeat(te,nlinks)
    table=detection_baselines(X[ti],y[ti],X[vi],y[vi],X[te],y[te],models)
    diagnosis={}
    lprobs=[]
    for i,f in enumerate(FAULTS):
        p=models.locators[f].predict_proba(LX[testlinks])[:,1]
        pred=p>=models.fault_threshold[f]
        pr,rc,f1,_=precision_recall_fscore_support(LY[testlinks,i],pred,
                                                    average='binary',zero_division=0)
        diagnosis[f]={'precision':float(pr),'recall':float(rc),'F1':float(f1),
                      'positives_test':int(LY[testlinks,i].sum())}
        lprobs.append(p)
    prob=np.stack(lprobs,axis=1)
    true=LY[testlinks]
    diag_pred=np.column_stack([prob[:,i]>=models.fault_threshold[f]
                               for i,f in enumerate(FAULTS)])
    diagnosis['micro_F1']=float(f1_score(true,diag_pred,average='micro',zero_division=0))
    diagnosis['macro_F1']=float(f1_score(true,diag_pred,average='macro',zero_division=0))
    # Component-localisation accuracy is a separate task from fault-label F1.
    # Every global time-row maps to exactly nlinks component rows.
    per_component=prob.max(axis=1).reshape(-1,nlinks)
    affected=true.any(axis=1).reshape(-1,nlinks)
    valid=affected.any(axis=1)
    ordered=np.argsort(-per_component[valid],axis=1)
    true_hits=affected[valid]
    diagnosis['component_top1_accuracy']=float(np.mean(
        [t[o[0]] for t,o in zip(true_hits,ordered)])) if valid.any() else None
    diagnosis['component_top3_accuracy']=float(np.mean(
        [any(t[o[:3]]) for t,o in zip(true_hits,ordered)])) if valid.any() else None
    diagnosis['component_localisation_test_windows']=int(valid.sum())
    per_label_true=true.reshape(-1,nlinks,len(FAULTS)).any(axis=1)
    per_label_pred=diag_pred.reshape(-1,nlinks,len(FAULTS)).any(axis=1)
    concurrent=per_label_true.sum(axis=1)>=2
    if concurrent.any():
        diagnosis['concurrent_label_jaccard']=float(np.mean([
            (np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))
            for a,b in zip(per_label_true[concurrent],per_label_pred[concurrent])]))
    else: diagnosis['concurrent_label_jaccard']=None
    record={'type':'synthetic_digital_twin_episode_holdout','samples_train':int(ti.sum()),
      'samples_validation':int(vi.sum()),'samples_test':int(te.sum()),
      'binary_detection':table,'diagnosis':diagnosis,
      'detector_validation_threshold':models.threshold,
      'diagnosis_validation_thresholds':models.fault_threshold,
      'test_split':'episode-disjoint, same 12-switch topology'}
    with open(os.path.join(base,'results','detection_metrics.json'),'w') as fp:
        json.dump(record,fp,indent=2)
    return record

