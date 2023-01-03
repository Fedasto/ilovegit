#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:30:33 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

distG=norm(loc=0, scale=0.1)


data=[distG.rvs(10000) for i in range(100)]
mean=[]

for j in data:
    # plt.figure()
    #plt.hist(j, bins=50, histtype='step', density=True)
    mean.append(np.mean(j))

plt.figure()
scale=np.std(mean)
xx=np.linspace(-scale*5,scale*5,10000)
k=3

# plt.scatter(xx,mean, marker='x', color='black', label='means')
plt.hist(mean, bins=10, histtype='step', color='black', label='means', density=True)
plt.plot(xx,t.pdf(xx, scale=scale, loc=0, df=k), color='red', label='t-distribution')
# plt.xlim(np.min(mean),np.max(mean))
plt.plot(xx,norm.pdf(xx,loc=0,scale=scale), label='gaussian')
plt.legend(loc=0)

