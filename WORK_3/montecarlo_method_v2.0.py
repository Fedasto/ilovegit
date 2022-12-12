# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 18:02:41 2022

@author: Federico Astori
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import uniform
from tqdm.auto import tqdm



integranda = lambda x, s: x**3 *np.exp(-((x**2)/(2*s**2)))*uniform.pdf(x) # my function times a uniform one
 # definisco la funzione integranda
 
sigma=np.random.random_integers(0.1,10,1)
N=[1, 10, 100, 1000, 10000, 100000] # number of samples

i_1=0
err_i_1=0
inte_1=[]
err_inte_1=[]
integrals_1=0
err_integrals_1=0
sigma_integrals_1=0

n_sample_integral=[]
n_sample_integrals_1_err=[]

for temp in tqdm(range(len(N))): # In this loop I check the dependece of the integral on the sample dimension

    for i in range(N[temp]): 
        i_1, err_i_1=(quad(integranda,0,np.infty, args=(sigma,))) #integrates from 0 to infinity
        inte_1.append(i_1)
        err_inte_1.append(err_i_1)
    
    integrals_1, err_integrals_1 = np.mean(inte_1), np.mean(err_inte_1)
    
    sigma_integrals_1 = np.std(inte_1)/np.sqrt(N)
    
    n_sample_integral.append(integrals_1)
    n_sample_integrals_1_err.append(err_integrals_1)
    
plt.figure(figsize=[5,5])   
plt.xscale('log')    
plt.scatter(N, n_sample_integral)
plt.xticks(N, N)
plt.xlabel('# samples')
plt.ylabel('integral result')
plt.ylim(np.min(n_sample_integral)-0.6,np.max(n_sample_integral)+0.6 )
plt.grid(True)


plt.figure(figsize=[5,5])   
plt.xscale('log')    
plt.scatter(N, n_sample_integrals_1_err)
plt.xticks(N, N)
plt.xlabel('# samples')
plt.ylabel('error on the integral result')
plt.ylim(np.min(n_sample_integrals_1_err)-0.1e-5,np.min(n_sample_integrals_1_err)+0.1e-5 )
plt.grid(True)

'''plt.figure(figsize=[8,8])
plt.bar(N, n_sample_integral, bottom=0.248973570997237, width=1.)'''

