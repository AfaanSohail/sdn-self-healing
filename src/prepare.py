from pathlib import Path
import numpy as np
import json
from dataset import collect_development
from models import FaultModels
from analysis import save_detection

if __name__=='__main__':
 b=Path(__file__).resolve().parents[1]
 data=collect_development(b/'results/split_manifest.csv')
 X,y,LX,LY,splits,n=data
 np.savez_compressed(b/'data/development.npz',X=X,y=y,LX=LX,LY=LY,splits=splits,nlinks=n)
 tr=splits=='train';va=splits=='validation'
 model=FaultModels().fit(X[tr],y[tr],LX[np.repeat(tr,n)],LY[np.repeat(tr,n)],X[va],y[va],LX[np.repeat(va,n)],LY[np.repeat(va,n)])
 model.save(b/'checkpoints/lightgbm_models.pkl')
 save_detection(str(b),model,data)
 import pickle
 with open(b/'results/detector_curves.pkl','wb') as f:pickle.dump(model.baseline_scores,f)
 X2=X.copy();L2=LX.copy();X2[:,16:23]=0;L2[:,18:20]=0
 m=FaultModels().fit(X2[tr],y[tr],L2[np.repeat(tr,n)],LY[np.repeat(tr,n)],X2[va],y[va],L2[np.repeat(va,n)],LY[np.repeat(va,n)])
 m.no_topology=True;m.save(b/'checkpoints/no_topology_models.pkl')
 print('DATA_AND_DETECTORS_DONE',flush=True)
