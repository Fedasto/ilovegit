#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:28:09 2022

@author: federicoastori
"""

import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import scipy.stats
from scipy.stats import norm
from scipy.stats import uniform
from scipy import optimize
from astroML import stats as astroMLstats

data = np.load('/Users/federicoastori/Desktop/ilovegit/WORK_6/formationchannels.npy')



xg=np.linspace(0,50,100)
#data = data[np.newaxis,:]
plt.title('two modes are clearly visible')
plt.hist(data, bins=100, fill=False, density=True, label='dataset')
plt.plot(xg,norm.pdf(xg,loc=20, scale=3), color='red')
plt.plot(xg,norm.pdf(xg,loc=30,scale=6), color='red', label='by-eye-fit')
plt.legend()


'''plt.figure(figsize=[10,10])
plt.title('three modes are clearly visible')
plt.scatter(np.linspace(0,100,2950),data, marker='.', color='black')
plt.vlines(12,0,50)
plt.vlines(32,0,50)

#binned statistics

plt.figure(figsize=[10,10])
plt.scatter(np.linspace(0,12,200),data[0:200], marker='.', color='red')
plt.scatter(np.linspace(12,32,1400),data[201:1601], marker='.', color='blue')
plt.scatter(np.linspace(32,100,1349),data[1601:2950], marker='.', color='green')
plt.vlines(12,0,50, color='black')
plt.vlines(32,0,50, color='black')
'''

plt.figure(figsize=[10,10])
plt.title('two modes are clearly visible')
plt.scatter(np.linspace(0,100,2950),data, marker='.', color='black')
plt.vlines(12,0,50, color='red')
plt.vlines(32,0,50, color='red')

#%%

# Part 2

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


'''def gmm_bic_score(estimator, data):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(data)'''

estimator=GaussianMixture(n_components=2, random_state=0).fit(data)

def aic_score(estimator,data):
    return -estimator.aic(data)



param_grid = {"n_components": range(1, 7), "covariance_type": ["spherical", "tied", "diag", "full"]}
grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=aic_score)
grid_search.fit(data)




