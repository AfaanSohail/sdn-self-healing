"""Active-policy comparisons; passive B0 stays in the numeric context table."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report_plot_style import apply
apply()

def render_recovery_figures():
 r=Path(__file__).resolve().parents[1]/'results';f=r/'figures'
 summary=json.loads((r/'summary.json').read_text());df=pd.read_csv(r/'episodes.csv')
 methods=['B1_rule','B2_greedy','B3_GA','B4_PSO','Full_DQN'];labels=['B1 Shortest','B2 Greedy','B3 GA','B4 PSO','Full DQN']
 fig,axes=plt.subplots(1,2,figsize=(9.5,3.6))
 for ax,n in zip(axes,(12,20)):
  rows={x['method']:x for x in summary if x['topology']==n};mean=np.array([100*rows[m]['sla_burden'] for m in methods]);ci=np.array([rows[m]['sla_burden_CI95'] for m in methods]).T*100
  ax.bar(range(5),mean,color=plt.cm.tab10(np.arange(5)))
  ax.errorbar(range(5),mean,yerr=np.maximum(0,np.vstack([mean-ci[0],ci[1]-mean])),fmt='none',color='black',capsize=3)
  ax.set(xticks=range(5),xticklabels=labels,ylabel='SLA burden (%)',title=f'{n} switches: active policies');ax.tick_params(axis='x',rotation=25)
 fig.tight_layout();fig.savefig(f/'sla_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
 scenarios=['outage','congestion','loss','delay','concurrent','correlated','cascading','switch']
 grids=[df[(df.topology==n)&(df.feasible==1)].groupby(['scenario','method']).sla_burden.mean().unstack().reindex(scenarios)[methods].to_numpy()*100 for n in (12,20)]
 vmax=max(float(np.nanmax(a)) for a in grids)
 fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,n,a in zip(axes,(12,20),grids):
  im=ax.imshow(a,aspect='auto',vmin=0,vmax=vmax,cmap='YlOrRd');ax.set(xticks=range(5),xticklabels=labels,yticks=range(8),yticklabels=scenarios,title=f'{n} switches');ax.tick_params(axis='x',rotation=35)
  for i in range(8):
   for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>.7*vmax else 'black')
 fig.colorbar(im,ax=axes,label='SLA burden (%)',fraction=.035,pad=.02)
 fig.savefig(f/'scenario_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':render_recovery_figures()
