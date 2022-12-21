#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:01:17 2022

@author: federicoastori
"""
# Part 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



N=5 # number of measurments
sigma=norm.rvs(loc=0.2, scale=0.05, size=N)
mean=1.0

data = np.concatenate([norm(mean,s).rvs(1) for s in sigma]) # mean=1 and error=sigma

plt.hist(data, density=True, fill=False)


x_grid=np.linspace(0,2,5000)
gaussian_Likelihoods=np.empty([len(data),5000])
                   
gaussian_Likelihoods = np.array([norm.pdf(x_grid,loc=s,scale=ss) for s,ss in zip(data, sigma)])

product_L= np.prod(gaussian_Likelihoods, axis=0)
#print(product_L)

plt.figure(figsize=[8,8])
for i in range(N):
    plt.title('MLE gaussian')
    plt.xlabel('$\mu$')
    plt.ylabel(r'$P(x_i|\mu,\sigma)$')
    plt.plot(x_grid, gaussian_Likelihoods[i], label='Likelihood # %1.0f' % (i) )
    plt.legend(loc='best')

plt.plot(x_grid, product_L,'--', label='product of likelihhods',c='black')  
plt.legend()

mle_solution=x_grid[np.argsort(product_L)[-1]]
mle_estimator=np.average(data, weights=1/sigma**2)

print('Likelihood is maximaized at %1.3f \n My estimator is the mean %1.3f ' % (mle_solution, mle_estimator))



fisher_marix=np.diff(np.log(product_L), n=2) # 2nd oreder derivative f the log liklihood
sigma_mu=((-1*fisher_marix)/((x_grid[1]-x_grid[0])**2))**(-0.5) # d^2mu = (x_grid[1]-x_grid[0])**2
sigma_mean=np.sum(sigma**-2)**-0.5 #sigma/np.sqrt(N) homoscidsics, but here are heteroscidesics look a the first 3 lectures

print('uncertianity on the mean is = %1.3f \n fisehr matrix formula =  %1.3f' % (sigma_mean, sigma_mu[0]) )

C=2.55
plt.figure(figsize=[8,8])
plt.plot(x_grid, C*norm.pdf(x_grid, loc=mle_estimator, scale=sigma_mean),'--' ,c='red',label=r'$L_\mathrm{fit}(\{x\})$') # plot of fitted gaussian C = normalization factor
plt.plot(x_grid, product_L, label=r'$L(\{x\})$', c='black') # plot of the numerial likelihood
plt.title('Comparison between numerical likelihood and fitted gaussian')
plt.legend()