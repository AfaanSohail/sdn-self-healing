"""Masked DQN with 128/64 MLP, replay buffer and target network."""
import collections
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Small models run much faster with a fixed thread count on shared CPUs.
torch.set_num_threads(1)

class QNetwork(nn.Module):
    def __init__(self, size=30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size,128),nn.ReLU(),
                                 nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x): return self.net(x)

class DQN:
    def __init__(self,seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.rng=random.Random(seed)
        self.network=QNetwork();self.target=copy.deepcopy(self.network)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=.001)
        self.memory=collections.deque(maxlen=50_000)
        self.learn_steps=0

    def choose(self,state,mask,eps=0.):
        legal=[i for i,ok in enumerate(mask) if ok]
        if not legal: return 0
        if self.rng.random() < eps: return self.rng.choice(legal)
        with torch.no_grad():
            q=self.network(torch.tensor(state,dtype=torch.float32).unsqueeze(0))[0].numpy()
        q[~np.asarray(mask,dtype=bool)] = -1e9
        return int(np.argmax(q))

    def remember(self,st,a,reward,st2,mask2,done):
        self.memory.append((st,a,reward,st2,np.array(mask2,dtype=bool),float(done)))

    def update(self):
        if len(self.memory)<64: return
        batch=self.rng.sample(self.memory,64)
        s=torch.tensor(np.stack([b[0] for b in batch]),dtype=torch.float32)
        a=torch.tensor([b[1] for b in batch],dtype=torch.int64).unsqueeze(1)
        r=torch.tensor([b[2] for b in batch],dtype=torch.float32)
        ns=torch.tensor(np.stack([b[3] for b in batch]),dtype=torch.float32)
        nm=torch.tensor(np.stack([b[4] for b in batch]),dtype=torch.bool)
        done=torch.tensor([b[5] for b in batch],dtype=torch.float32)
        with torch.no_grad():
            future=self.target(ns).masked_fill(~nm,-1e9).max(1).values
            future=torch.where(nm.any(dim=1),future,torch.zeros_like(future))
            qtarget=r+.99*(1-done)*future
        pred=self.network(s).gather(1,a).squeeze(1)
        loss=F.smooth_l1_loss(pred,qtarget)
        self.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),5.)
        self.optimizer.step()
        self.learn_steps+=1
        if self.learn_steps%500 ==0:
            self.target.load_state_dict(self.network.state_dict())

    def save(self,path):
        torch.save({'network':self.network.state_dict(),'learn_steps':self.learn_steps},path)
    @classmethod
    def load(cls,path):
        instance=cls(0)
        package=torch.load(path,map_location='cpu',weights_only=True)
        instance.network.load_state_dict(package['network']);instance.network.eval()
        return instance
