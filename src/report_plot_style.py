"""Consistent Times New Roman black labels for report figures."""
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib import font_manager
import numpy as np

def apply():
 folder=os.environ.get('PHASE3_FONT_DIR')
 if folder:
  for path in Path(folder).glob('*.TTF'):font_manager.fontManager.addfont(str(path))
 mpl.rcParams.update({'font.family':'Times New Roman','text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black','xtick.color':'black','ytick.color':'black'})
 original=Figure.savefig
 if getattr(original,'_phase3_style',False):return
 def save(self,*args,**kwargs):
  for text in self.findobj(match=Text):text.set_fontfamily('Times New Roman');text.set_color('black')
  for ax in self.axes:
   for im in ax.images:
    data=np.asarray(im.get_array())
    if data.shape==(2,2):im.set_cmap('Blues');im.set_clim(0,float(np.max(data))*2)
  return original(self,*args,**kwargs)
 save._phase3_style=True;Figure.savefig=save
