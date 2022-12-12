# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:29:00 2022

@author: PC
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

n_death=np.arange(0,5,1)
n_groups=[109,65,22,3,1]
tot_groups=np.sum(n_groups)


mu=np.arange(0,2,0.35)

plt.figure(figsize=[8,8])
for i in range(len(mu)):
    plt.scatter(n_death, n_groups/tot_groups, c='black', marker='o')
    plt.plot(n_death,poisson(mu[i]).pmf(n_death), label='$\mu=%1.1f$' % (mu[i])) # PMF probability mass function, where mu is a shape parameter
    plt.legend(loc='best')
best_mu=np.average(n_death, weights=n_groups/tot_groups) #weighted mean along an axis

plt.figure(figsize=[8,8])
plt.scatter(n_death, n_groups/tot_groups, c='black', marker='o')
plt.plot(n_death,poisson(best_mu).pmf(n_death), color='red', label='best $\mu=%1.1f$' % (best_mu))
plt.legend(loc='best')
