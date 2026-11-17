"""Official SpotLight platform SAMPLE; chronological supplementary detection study.
No claim of full-dataset RANGAN replication. Labels never enter feature columns.
"""
from pathlib import Path
import hashlib,json,pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score,average_precision_score,confusion_matrix
from lightgbm import LGBMClassifier

if __name__=='__main__':
 b=Path(__file__).resolve().parents[1];folder=b/'data/spotlight'
 raw=pd.read_csv(folder/'platform.csv');raw['dt']=pd.to_datetime(raw.timestamp,errors='coerce')
 df=raw[raw.dt.notna()].drop_duplicates('timestamp').sort_values('dt').copy()
 intervals=np.unique(np.loadtxt(folder/'label',delimiter=',',dtype=np.int64),axis=0)
 # Retain annotated span plus 30 seconds of clean follow-up, set before fitting.
 df=df[df.dt.astype('int64')<=intervals.max()+30_000_000_000].reset_index(drop=True)
 y=np.zeros(len(df),int);ts=df.dt.astype('int64').to_numpy()
 for lo,hi in intervals:y|=(ts>=lo)&(ts<=hi)
 x=df.drop(columns=['dt','timestamp','Unnamed: 0'],errors='ignore').apply(pd.to_numeric,errors='coerce')
 n=len(x);a=int(.6*n);c=int(.8*n);gap=10
 tr=np.arange(a);va=np.arange(a+gap,c);te=np.arange(c+gap,n)
 keep=x.iloc[tr].notna().any();x=x.loc[:,keep]
 # Trailing 5-sample mean; causal context may precede a partition but labels do not.
 x=x.rolling(5,min_periods=1).mean()
 imputer=SimpleImputer(strategy='median',add_indicator=True)
 X=imputer.fit_transform(x.iloc[tr]);V=imputer.transform(x.iloc[va]);T=imputer.transform(x.iloc[te])
 models={'LightGBM':LGBMClassifier(n_estimators=100,num_leaves=15,learning_rate=.05,class_weight='balanced',n_jobs=1,verbosity=-1,random_state=11),
 'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=12,class_weight='balanced',n_jobs=1,random_state=11),
 'RBF SVM':make_pipeline(StandardScaler(),SVC(class_weight='balanced')),
 'Isolation Forest':make_pipeline(StandardScaler(),IsolationForest(n_estimators=100,random_state=11))}
 metrics={};curves={}
 for name,m in models.items():
  iso=name=='Isolation Forest'
  if iso:m.fit(X[y[tr]==0])
  else:m.fit(X,y[tr])
  def score(z):
   if iso:return -m.score_samples(z)
   if name=='RBF SVM':return m.decision_function(z)
   return m.predict_proba(z)[:,1]
  v=score(V);s=score(T)
  threshold=float(max(np.unique(np.quantile(v,np.linspace(0,1,101))),key=lambda t:f1_score(y[va],v>=t,zero_division=0)))
  pred=s>=threshold;target=y[te]
  metrics[name]={'accuracy':accuracy_score(target,pred),'precision':precision_score(target,pred,zero_division=0),'recall':recall_score(target,pred,zero_division=0),'F1':f1_score(target,pred,zero_division=0),'macro_F1':f1_score(target,pred,average='macro',zero_division=0),'ROC_AUC':roc_auc_score(target,s),'PR_AUC':average_precision_score(target,s),'threshold':threshold,'confusion_matrix':confusion_matrix(target,pred).tolist()}
  curves[name]=(target,s,pred)
  with open(b/f'checkpoints/spotlight_{name.replace(" ","_")}.pkl','wb') as f:pickle.dump(m,f)
 manifest={'source':'https://github.com/netsys-edinburgh/SpotLight/tree/main/sample','scope':'Official sample only; platform KPIs, single trace; NOT full SpotLight or RANGAN replication','raw_rows':len(raw),'retained_rows':n,'invalid_header_rows':int(raw.dt.isna().sum()),'window_samples':5,'gap_samples':gap,'numeric_features':int(keep.sum()),'split_counts':{k:{'rows':len(i),'positive':int(y[i].sum()),'first':str(df.dt.iloc[i[0]]),'last':str(df.dt.iloc[i[-1]])} for k,i in [('train',tr),('validation',va),('test',te)]},'metrics':metrics,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}}
 with open(b/'results/spotlight_metrics.json','w') as f:json.dump(manifest,f,indent=2)
 with open(b/'results/spotlight_curves.pkl','wb') as f:pickle.dump(curves,f)
 with open(b/'checkpoints/spotlight_preprocessor.pkl','wb') as f:pickle.dump({'imputer':imputer,'columns':x.columns.tolist()},f)
 df[['timestamp']].assign(label=y,split=['train' if i<a else 'validation' if a+gap<=i<c else 'test' if i>=c+gap else 'gap' for i in range(n)]).to_csv(b/'results/spotlight_split_manifest.csv',index=False)
 print(json.dumps(manifest,indent=2))
