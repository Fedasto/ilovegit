#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 14:37:54 2022

@author: federicoastori
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from astroML import stats as astroMLstats

#from astroML.plotting import setup_text_plots
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

'''setup_text_plots(fontsize=14, usetex=True)
%config InlineBackend.figure_format='retina' # very useful command for high-res images'''


data = np.load('/Users/federicoastori/Desktop/ilovegit/WORK_6/formationchannels.npy') # masses of BHs measured by LIGO to train


plt.figure(figsize=[10,10])
xg=np.linspace(0,50,100)
#data = data[np.newaxis,:]
plt.title('3 modes are clearly visible') # the 10% of the total mass are in binary, alone BH instead have the 6% (more or less)
plt.hist(data, bins=100, fill=False, density=True, label='dataset')
plt.plot(xg,norm.pdf(xg,loc=norm.fit(data[:200,0])[0],scale=norm.fit(data[:200,0])[1]), color='red') # loc=np.mean(data[:200]), scale=np.std(data[:200])
plt.plot(xg,norm.pdf(xg,loc=np.median(data[201:1470]), scale=astroMLstats.sigmaG(data[201:1470])), color='red')
plt.plot(xg,norm.pdf(xg,loc=np.mean(data[1471:]),scale=np.std(data[1475:])), color='red', label='by-eye-fit')
plt.xlabel('BH masses [$M_\odot$]')



xgrid=np.linspace(0,50, 2950)
AIC=[]
BIC=[]


for i in range(1,10,1):
    gmm = GaussianMixture(n_components=i, n_init=20, random_state=0).fit(data)
    aic = gmm.aic(data)
    bic=gmm.bic(data)
    logprob = gmm.score_samples(xgrid.reshape(-1, 1))
    fx = lambda j : np.exp(gmm.score_samples(j.reshape(-1, 1)))
    plt.plot(xgrid, fx(np.array(xgrid)), '-', color='C%1.0f' % (i),label="$f(x)$, parametric (%1.0f Gaussians)" % (i))
    plt.legend(loc='best')
    
    #print(fx(np.median(data[:1470])), fx(np.mean(data[1471:])))
    AIC.append(aic)
    BIC.append(bic)


plt.figure(figsize=[5,5])
#plt.bar(np.arange(1,10,1),AIC,fill=False)
plt.scatter(np.arange(1,10,1),AIC, marker='x', color='Black', label='AIC')
plt.scatter(np.arange(1,10,1),BIC, marker='x', color='blue', label='BIC')
plt.ylim(19250,20000)
plt.xlim(0,10)
plt.legend(loc='best')
# plt.hlines(np.max(AIC),1,10, color='red')


best_aic_index=np.argmin(AIC)+1 # n_components has to be the index plus 1
        
#plt.bar(best_aic_index,np.min(AIC),color='green', alpha=0.5)
plt.title('less is more')
plt.xlabel('n_component')
plt.ylabel('Information criterion')
print(np.min(AIC))

#%%

# Part 2

# colors = ["navy", "turquoise", "cornflowerblue", "darkorange"]

plt.figure(figsize=[8,8])
plt.hist(data, bins=30, fill=False, density=True, histtype='step', color='b', label='dataset')

gmm_best = GaussianMixture(n_components=best_aic_index, random_state=0, n_init=20).fit(data)

fx = lambda j : np.exp(gmm_best.score_samples(j.reshape(-1, 1))) # PDF

plt.plot(xgrid,fx(np.array(xgrid)), color='red', label='best model')

#########

predict_pdf = gmm_best.predict_proba(xgrid.reshape(-1,1))
single_pdf = predict_pdf * fx(xgrid)[:, np.newaxis]

for i in range(best_aic_index):
    plt.plot(xgrid, single_pdf[:,i],'--', c='C0', label='single mode %1.0f' % (i) )


plt.xlabel("Black hole mass $[M_\odot]$")
plt.legend(loc='best')
# plt.scatter(predict_pdf[0:-1],predict_pdf[1:], marker='x', color='black')


