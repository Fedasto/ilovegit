# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:59:25 2022

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics




means=[]
err_means=[]
stds=[]
err_stds=[]
n=50

for temp in range(n):
    plt.figure(figsize=[8,8])
    N=1000 # number of data points
    
    x=np.random.normal(1,10,N)
    
    sample_mean = np.mean(x)
    sample_std = np.sqrt(1/(N-1)*(np.sum((x-sample_mean)**2))) # tako bessel's correction
    
    err_sample_mean = sample_std/np.sqrt(N)
    err_sample_std = sample_std/(np.sqrt(2*(N-1)))
    
    q25, q50, q75 = np.percentile(x,[25,50,75])
    sigma_G=0.7413*(q75-q25) # interquantile range
    
    #h75,h25= 
    #err_sigma_G = 0.7143*(1/h75*np.sqrt((0.75*(1-0.75))/N) - (1/h25*np.sqrt((0.25*(1-0.25))/N)))
    
    means.append(sample_mean)
    err_means.append(err_sample_mean)
    
    stds.append(sample_std)
    err_stds.append(err_sample_std)
    
    plt.hist(x, bins=20, fill=False, density=True)
    x_grid=np.linspace(-35,35,N)
    plt.plot(x_grid, norm.pdf(x_grid, sample_mean, sample_std), label='best estimate')
    plt.plot(x_grid, norm.pdf(x_grid, sample_mean+err_sample_mean, sample_std+err_sample_std), linestyle='--', label=' sample + error')
    plt.plot(x_grid, norm.pdf(x_grid, sample_mean-err_sample_mean, sample_std-err_sample_std),linestyle='-.', label='sample - error')
    plt.legend(loc='best')
    plt.show()

plt.figure(figsize=[8,8])
plt.scatter(means,stds, c='black', s=5)
plt.errorbar(means,stds,yerr=err_stds, xerr=err_means, capsize=2, fmt='o', c='black')
plt.xlabel('mean of the samples')
plt.ylabel('std of the samples')

plt.figure(figsize=[8,8])
plt.bar(np.arange(0,n), means, width=2, color='black', fill=False, yerr=err_means, capsize=2)
plt.ylabel('sample means')

plt.figure(figsize=[8,8])
plt.bar(np.arange(0,n),  stds, width=2, color='black', fill=False, yerr=err_stds, capsize=2)
plt.ylabel('sample stds')
