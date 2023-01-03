#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 15:18:33 2022

@author: federicoastori
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

# posterior = (prior x likelihood=AIC) / (marginals=gmm)

data=np.load('/Users/federicoastori/Desktop/ilovegit/WORK_6/formationchannels.npy')

N=np.arange(1,10)
gmm=[]
AIC=[]

for iteration in range(10):
    for temp in range(len(N)):
        gmm.append(GaussianMixture(n_components=N[temp],n_init=20).fit(data))
        AIC.append(gmm[temp].aic(data))
        
    print('iteration ' ,iteration , 'best model N= ', N[np.argmin(AIC)])