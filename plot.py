import numpy as np
from matplotlib import pyplot as plt
from _datetime import datetime as dt
ax=[2,3,4,5,6]
a1y=np.array([26.56,15.90,10.81,8.94,7.33])/5*4/3
a2y=np.array([33.18,18.29,12.93,10.48,8.77])/5*4/3
a3y=np.array([19.0316,11.2543,7.7747,6.2926,5.4713])/3
linewidth,markersize=1,8
plt.plot(ax, a1y, c="b", marker='*',linewidth=linewidth,markersize=markersize,label='Baseline scheme 2')
plt.plot(ax, a2y, c="g", marker='^',linewidth=linewidth,markersize=6,label='Baseline Scheme 1')
plt.plot(ax, a3y, c="r", marker='d',linewidth=linewidth,markersize=6,label='Proposed Scheme')
plt.legend()
plt.grid(color = 'lightskyblue', linestyle = '--', linewidth = 0.5)
plt.xlabel('Num of Interference UAVs')
plt.ylabel('Average SINR of the Targets ')
timestamp = dt.strftime(dt.now(), '%Y_%m_%d_%Hh%Mm%Ss')
plt.savefig('figs/performances_' + timestamp + '.png', dpi=1000)