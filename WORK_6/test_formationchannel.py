#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 17:35:10 2022

@author: federicoastori
"""
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from astroML import stats as astroMLstats
from matplotlib.patches import Ellipse
from scipy import linalg
#from astroML.plotting import setup_text_plots
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

'''setup_text_plots(fontsize=14, usetex=True)
%config InlineBackend.figure_format='retina' # very useful command for high-res images'''


data = np.load('/Users/federicoastori/Desktop/ilovegit/WORK_6/formationchannels.npy') # masses of BHs measured by LIGO to train


plt.figure(figsize=[10,10])
xg=np.linspace(0,50,100)
#data = data[np.newaxis,:]
plt.title('Two modes are clearly visible') # the 10% of the total mass are in binary, alone BH instead have the 6% (more or less)
plt.hist(data, bins=100, fill=False, density=True, label='dataset')
plt.plot(xg,norm.pdf(xg,loc=np.median(data[:1470]), scale=astroMLstats.sigmaG(data[:1470])), color='red')
plt.plot(xg,norm.pdf(xg,loc=np.mean(data[1471:]),scale=np.std(data[1475:])), color='red', label='by-eye-fit')
plt.xlabel('number of BH')
plt.ylabel('masses')


xgrid=np.linspace(0,50, 1000)
AIC=[0]


for i in range(2,10,1):
    gmm = GaussianMixture(n_components=i, random_state=0).fit(data)
    aic = gmm.aic(data)
    logprob = gmm.score_samples(xgrid.reshape(-1, 1))
    fx = lambda j : np.exp(gmm.score_samples(j.reshape(-1, 1)))
    plt.plot(xgrid, fx(np.array(xgrid)), '-', color='C%1.0f' % (i),label="$f(x)$, parametric (%1.0f Gaussians)" % (i))
    plt.legend(loc='best')
    
    #print(fx(np.median(data[:1470])), fx(np.mean(data[1471:])))
    AIC.append(aic)

plt.figure(figsize=[10,10])
plt.bar(np.arange(1,10,1),AIC,fill=False)
plt.ylim(19250,20000)
plt.xlim(0,10)
plt.hlines(np.max(AIC),1,10, color='red')

for i in range(len(AIC)):
    if AIC[i]==np.max(AIC):
        best_aic_index=i+1
        
plt.bar(best_aic_index,np.max(AIC),color='green', alpha=0.5)
plt.title('less is more')
print(np.max(AIC))

#%%

# Part 2

colors = ["navy", "turquoise", "cornflowerblue", "darkorange"]


plt.hist(data, bins=100, fill=False, density=True)
clf= GaussianMixture(n_components=best_aic_index, random_state=0)
gm = clf.fit(data)
fx = lambda j : np.exp(gm.score_samples(j.reshape(-1, 1)))
plt.plot(xgrid,fx(np.array(xgrid)), color='red')

'''plt.figure(figsize=[8,8])
predict_pdf = gm.predict_proba(data)
plt.scatter(predict_pdf[0:-1],predict_pdf[1:], marker='x', color='black')
'''

'''plt.figure(figsize=[8,8])
x = np.array(np.linspace(0,50,2950)).reshape(-1,1)
y = clf.score_samples(x)

plt.plot(x, y)'''









