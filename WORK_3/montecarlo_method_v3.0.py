# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 15:15:53 2022

@author: PC
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm.auto import tqdm

#N=10000
#sigma=2

def integral(N,sigma):
    global I,knownresult, abs_err
    #x=np.random.normal(0,sigma,N)
    #x_grid=np.linspace(np.min(x)-2,np.max(x)+2,N)
    
    #plt.figure(figsize=[8,8])
    #plt.hist(x, bins=50, fill=False, density=True)
    #plt.plot(x_grid, norm(loc=0,scale=sigma).pdf(x_grid), color='red', label='gaussian') # PDF=prob density function
    #plt.legend(loc='best')
    
    #plt.figure(figsize=[8,8])
    #plt.hist(np.abs(x), bins=50, fill=False, density=True)
    #plt.plot(np.abs(x_grid), np.abs(2*norm(loc=0,scale=sigma).pdf(x_grid)), color='red', label='half gaussian') # it requires the 2 in order to satisfy the double counting
    
    samples = np.abs(norm(loc=0,scale=sigma).rvs(N)) #RVS=randomvariates
    I = sigma*(np.pi/2)**0.5 * np.mean(samples**3) # montecarlo application
    knownresult= 2*sigma**4 
    abs_err=np.abs(I-knownresult)/knownresult
    
    #print(r'Montecarlo result = %1.3f vs known result $2\sigma^{4}$ = %1.3f ' % (I,knownresult), 'absolute error= ', abs_err)
    return (I, knownresult, abs_err  )

sigma=int(np.random.random_integers(0,100,1)) 
N=[1e3,2e3,3e3,5e3,7e3,1e4, 3e4, 5e4,8e4,1e5, 5e5, 9e5,1e6,1e7,1e8]#np.linspace(10,1e5,5)

def distribution(N,sigma):
    global matrix
    
    result=[]
    knownres=[]
    error=[]
       
    for temp in range(len(N)):
        integral(int(N[temp]),sigma)
        result.append(I)
        knownres.append(knownresult)
        error.append(abs_err)
    
    '''  
    plt.figure(figsize=[8,8])
    plt.xscale('log')
    plt.scatter(N,result, color='black')
    plt.plot(N,result, color='black', linestyle='--')
    plt.hlines(knownres,np.min(N),np.max(N), color='red', label=r'$2\sigma^{4}$')
    plt.legend(loc='best')  
    plt.title('result of the integral with MC method')  
    
    x_grid=np.linspace(990,1e8,100000)
    plt.figure(figsize=[8,8])
    plt.xscale('log')
    plt.scatter(N,error, color='black')
    plt.plot(N,error, color='black', linestyle='--')
    plt.plot(x_grid,1/np.sqrt(x_grid),color='red', label=r'$frac{1}{\sqrt{N}}$')
    plt.title('integral absolute error')
    plt.legend(loc='best')'''
        
    matrix=[result,knownres,error]

    return (matrix)

realization=[100,1000,10000]
R=[]

for temp in tqdm(range(len(realization))):
    distribution(N,sigma)
    R.append(matrix)

plt.figure(figsize=[5,5]) # le barre devon essere gaussiane
plt.bar(N[0], R[0][0], fill=False, log=True, width=N)# histo 10 realization
plt.xscale('log')
plt.title('%1.0f Realiztions integral result' % realization[0])
plt.figure(figsize=[5,5])
plt.bar(N, R[1][0], fill=False, log=True, width=N) # histo 100 realization
plt.xscale('log')
plt.title('%1.0f Realiztions integral result' % realization[1])
plt.figure(figsize=[5,5])
plt.bar(N, R[2][0],fill=False, log=True, width=N) # histo 1000 realization  
plt.xscale('log')
plt.title('%1.0f  Realiztions integral result' % realization[2])