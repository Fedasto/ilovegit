#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:13:45 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from astroML.stats import sigmaG

#%%
N=100000 # number of days
today=[1] # 1 means cloud day

'''prob_rain_given_rain = 0.5
prob_sun_given_rain = 0.5
prob_sun_given_sun = 0.9
prob_rain_given_rain=0.1'''

for i in tqdm(range(N)):
    if today[i]==1:
        tomorrow=np.random.randint(0,1,1)
        today.append(tomorrow)
        
    elif today[i]==0:
        tomorrow=np.random.randint(1,9,1)
        
        if tomorrow==1:
            today.append(1)
        else:
            today.append(0)
#%%            
x_grid=np.linspace(0,N,N-1) # number of days 

sunny_day=0
running_average=[]

for j in range(1,N):
    if today[j]==0:
        sunny_day +=1
        
    running_average.append(sunny_day/j)

plt.figure()
plt.title('Trace plot - sunny days')
plt.xlabel('day')
plt.ylabel('running average')    
plt.plot(x_grid, running_average, color='black')

plt.figure()
plt.hist(running_average,bins=350, density=True, histtype='step', label='sunny days')

xx = np.linspace(0.85,0.91, 1000)
plt.plot(xx, norm.pdf(xx, loc=np.median(running_average), scale=sigmaG(running_average)),'r-', label='norm pdf')
plt.legend(loc='upper left')
plt.xlim(0.881,0.896)


#%%
# if I start when the prob is stationary

burn_idx=1000



sunny_day=0
running_average=[]


for j in range(burn_idx,N):
    if today[j]==0:
        sunny_day +=1
        
    running_average.append(sunny_day/j)

plt.figure()
plt.hist(running_average,bins=300, density=True, histtype='step', label='sunny days')

xx = np.linspace(0.6,0.91, 1000)
plt.plot(xx, norm.pdf(xx, loc=np.median(running_average), scale=sigmaG(running_average)),'r-', label='norm pdf')
plt.legend(loc='upper left')
plt.xlim(0.75,0.9)


            

    


    