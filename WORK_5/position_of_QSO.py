#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:27:56 2022

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Part 1

N=5 # number of measurments
sigma=0.2
mean=1.0

data = norm.rvs(loc=mean, scale=sigma, size=N) # mean=1 and error=0.2

plt.hist(data, density=True, fill=False)

x_grid=np.linspace(0,2,5000)
gaussian_Likelihoods=np.empty([len(data),5000])
                   
for j in range(len(data)):
    for k in range(1,5000,1):
        gaussian_Likelihoods[j][k]=(norm(loc=mean,scale=sigma).pdf(x_grid[k]))
        gaussian_Likelihoods[j][0]=j
    
'''gaussian_Likelihoods = np.array([norm.pdf(xgrid,loc=s,scale=sigma) for s in sample])''' # easier way to make this kind of for-cycle

product_L= np.prod(gaussian_Likelihoods, axis=0)
#print(product_L)

plt.figure(figsize=[8,8])
for i in range(N):
    plt.title('MLE gaussian')
    plt.xlabel('$\mu$')
    plt.ylabel(r'$P(x_i|\mu,\sigma)$')
    plt.plot(x_grid, norm.pdf(x_grid, loc=data[i], scale=sigma) , label='Likelihood # %1.0f' % (i) ) #
    plt.legend(loc='best')

plt.plot(x_grid, product_L,'--', label='product of likelihhods',c='black')  
plt.legend()

mle_solution=x_grid[np.argsort(product_L)[-1]]
mle_estimator=np.mean(data)

print('Likelihood is maximaized at %1.3f \n My estimator is the mean %1.3f ' % (mle_solution, mle_estimator))

#%% 
# Part 2

fisher_marix=np.diff(np.log(product_L), n=2) # 2nd oreder derivative f the log liklihood
sigma_mu=((-1*fisher_marix)/((x_grid[1]-x_grid[0])**2))**(-0.5) # d^2mu = (x_grid[1]-x_grid[0])**2
sigma_mean=sigma/np.sqrt(N)

print('uncertianity on the mean is = %1.3f \n fisehr matrix formula =  %1.3f' % (sigma_mean, sigma_mu[1]) )

C=7
plt.figure(figsize=[8,8])
plt.plot(x_grid, C*norm.pdf(x_grid, loc=mle_estimator, scale=sigma_mean),'--' ,c='red',label=r'$L_\mathrm{fit}(\{x\})$') # plot of fitted gaussian C = normalization factor
plt.plot(x_grid, product_L, label=r'$L(\{x\})$', c='black') # plot of the numerial likelihood
plt.title('Comparison between numerical likelihood and fitted gaussian')
plt.legend()




  
    
    

