"""LightGBM binary detector + shared per-link multi-label diagnosis."""
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score
from features import global_features, link_features, GLOBAL_FEATURES, LINK_FEATURES, all_link_features
from config import FAULTS

class FaultModels:
    def __init__(self):
        self.detector = None
        self.locators = {}
        self.threshold = .5
        self.fault_threshold = {f:.5 for f in FAULTS}
        self.normal_median = None
        self.normal_scale = None
        self.unknown_radius = 1e9

    def fit(self, X, y, LX, LY, XV, yv, LVX, LVY):
        self.detector = LGBMClassifier(n_estimators=100,num_leaves=15,
            learning_rate=.05, class_weight='balanced',verbosity=-1,n_jobs=2,
            random_state=11)
        self.detector.fit(X,y)
        valid_scores = self.detector.predict_proba(XV)[:,1]
        # False-alarm constraint checked on a *validation-only* quiet subset.
        quiet = (yv==0)
        choices = np.linspace(.3,.99,70)
        viable = [t for t in choices if np.sum(valid_scores[quiet]>=t) <= max(1,quiet.sum()//600)]
        if viable:
            self.threshold = float(max(viable, key=lambda t:f1_score(yv, valid_scores>=t,zero_division=0)))
        else:
            self.threshold = float(max(choices))
        for j,f in enumerate(FAULTS):
            if LY[:,j].sum() == 0:
                continue
            mdl = LGBMClassifier(n_estimators=100,num_leaves=15,
                learning_rate=.05,class_weight='balanced',verbosity=-1,n_jobs=2,
                random_state=31+j)
            mdl.fit(LX,LY[:,j])
            self.locators[f] = mdl
            scores = mdl.predict_proba(LVX)[:,1]
            self.fault_threshold[f] = float(max(np.linspace(.25,.8,23),
                key=lambda t:f1_score(LVY[:,j],scores>=t,zero_division=0)))
        normal = X[y==0]
        self.normal_median = np.median(normal,axis=0)
        self.normal_scale = np.maximum(np.std(normal,axis=0),.1)
        nd = np.sqrt(np.mean(((normal-self.normal_median)/self.normal_scale)**2,axis=1))
        self.unknown_radius = float(np.percentile(nd,99.9))
        return self

    def score(self, env):
        x = global_features(env.timeseries,env.flows).reshape(1,-1)
        if getattr(self,"no_topology",False): x[:,16:23]=0.
        fault_prob = float(self.detector.booster_.predict(x,num_threads=1)[0])
        return fault_prob

    def diagnose(self, env, max_links=8):
        LX = all_link_features(env.timeseries,env.edges)
        if getattr(self,"no_topology",False): LX[:,18:20]=0.
        probs = {label: mdl.booster_.predict(LX,num_threads=1)
                 for label,mdl in self.locators.items()}
        flagged = sorted([(float(max(probs[k][i] for k in probs)),e,i)
                          for i,e in enumerate(env.edges)],reverse=True) if probs else []
        class_probs = {k:float(v.max()) for k,v in probs.items()}
        top = [(e,{f:float(v[i]) for f,v in probs.items()}) for _,e,i in flagged[:max_links]]
        unknown = max(class_probs.values(),default=0.) < .30
        return class_probs, top, unknown

    def save(self,path):
        with open(path,'wb') as f: pickle.dump(self,f)
    @staticmethod
    def load(path):
        with open(path,'rb') as f: return pickle.load(f)


def detection_baselines(X,y,XV,yv,XT,yt,full):
    """Four matched detector baselines plus the proposed LightGBM head."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    # The fixed KPI baseline follows the *current sampled* threshold features;
    # it does not read labels, injected-fault schedules or future observations.
    def kpi(z):
        return ((z[:,1]>20) | (z[:,5]>.01) | (z[:,8]<.90)).astype(float)
    baselines = {
        'LightGBM': (full.detector,full.threshold,'proba'),
        'Fixed KPI thresholds': (None,.5,'fixed'),
        'Random Forest': (RandomForestClassifier(n_estimators=100,max_depth=12,
                           class_weight='balanced',n_jobs=2,random_state=21),None,'proba'),
        'RBF SVM': (make_pipeline(StandardScaler(),SVC(kernel='rbf',
                       class_weight='balanced',gamma='scale')),None,'margin'),
        'Isolation Forest': (make_pipeline(StandardScaler(),IsolationForest(n_estimators=100,contamination=.1,
                             n_jobs=2,random_state=21)),None,'isolation')}
    result={}
    for name,(model,threshold,kind) in baselines.items():
        if kind=='fixed':
            scores_valid,scores_test=kpi(XV),kpi(XT)
        else:
            if name!='LightGBM':
                model.fit(X[y==0] if kind=='isolation' else X,
                          **({} if kind=='isolation' else {'y':y}))
            def predict(z):
                if kind=='isolation':return -model.score_samples(z)
                if kind=='margin':return model.decision_function(z)
                return model.predict_proba(z)[:,1]
            scores_valid,scores_test=predict(XV),predict(XT)
        if threshold is None:
            # Val-only threshold to maintain a strict quiet-period false-alarm
            # constraint, even if that sacrifices sensitivity.
            threshold=float(max(np.unique(np.quantile(scores_valid,np.linspace(0,1,101))), key=lambda z:f1_score(yv,scores_valid>=z,zero_division=0)))
        yp=scores_test>=threshold
        p,r,f,_=precision_recall_fscore_support(yt,yp,average='binary',zero_division=0)
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
        if not hasattr(full,'baseline_scores'): full.baseline_scores={}
        full.baseline_scores[name]=(yt,scores_test,yp)
        result[name]={'threshold':float(threshold),'accuracy':float(accuracy_score(yt,yp)),'macro_F1':float(f1_score(yt,yp,average='macro')),'ROC_AUC':float(roc_auc_score(yt,scores_test)), 'confusion_matrix':confusion_matrix(yt,yp).tolist(),'precision':float(p),'recall':float(r),'F1':float(f),
                      'PR_AUC':float(average_precision_score(yt,scores_test)),
                      'false_positive_rows':int(np.sum(yp & (yt==0))),
                      'test_rows':len(yt)}
    return result
