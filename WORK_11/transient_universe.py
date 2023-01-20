#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:25:59 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import emcee

def burst_model(b,A,t0,alpha,t): # myposterior
    y=0
    if t.all()<t0:
        y==b # y=flux
    else:
        y=b+A*np.exp(-alpha*(t-t0))
    
    return y

data = np.load('/Users/federicoastori/Desktop/ilovegit/WORK_11/transient.npy')

# 1 column = time
# 2 column = flux
# 3 column = homoscedastic uncertianities on flux

plt.figure(figsize=(12,8))
plt.scatter(data[:,0],data[:,1], marker='o',color='black')
plt.errorbar(data[:,0],data[:,1],yerr=data[:,2], fmt='o', color='black', capsize=2)
plt.xlabel('time')
plt.ylabel('flux')
plt.title('Transient')

# burst model fit

b_par=np.random.uniform(0,50, 1)
A_par=np.random.uniform(0,50, 1)
t0_par=np.random.uniform(0,100, 1)
alpha_par=np.random.uniform(-5,5, 1)

xx=np.linspace(0,100,100)
plt.plot(xx ,burst_model(b_par,A_par,t0_par,alpha_par,xx))


# %%

# Note that since flat priors cancels out I don't wirte them: 1/50-0



def logL(b,A,t0,alpha, time, model=burst_model): #myloglikelihood
    """Gaussian log-ikelihood of the model at theta"""
    # theta=[b,A,t0,alpha]
    # time, flux, sigma_flux = data
    time = data[:,0]
    flux = data[:,1]
    sigma_flux = data[:,2]
    
    flux_fit = model(b,A,t0,alpha,time) # return the flux
    
    return sum(stats.norm.logpdf(args) for args in ((flux, flux_fit, sigma_flux)))

# pyMC sampler

ndim = 4  # number of parameters in the model
nwalkers = 20  # number of MCMC walkers
burn = 1000  # "burn-in" period to let chains stabilize
nsteps = 10000  # number of MCMC steps to take **for each walker**

# initialize parms

b_par=np.random.uniform(0,50, 1)
A_par=np.random.uniform(0,50, 1)
t0_par=np.random.uniform(0,100, 1)
alpha_par=np.random.uniform(-5,5, 1)


# initialize theta 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

# the function call where all the work happens: 
sampler = emcee.EnsembleSampler(nwalkers, ndim, logL, args=[b_par,A_par,t0_par,alpha_par])
sampler.run_mcmc(starting_guesses, nsteps)
 
# sampler.chain is of shape (nwalkers, nsteps, ndim)
# throw-out the burn-in points and reshape:
emcee_trace  = sampler.chain[:, burn:, :].reshape(-1, ndim)

print("done \n")
print('sampler chain shape \n',sampler.chain.shape) #original chain structure
print('emcee race shape',emcee_trace.shape) #burned and flattened chain

emcee_trace.flatten()

# plot 
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(15, 8))
#fig.subplots_adjust(left=0.11, right=0.95, 
                   # wspace=0.35, bottom=0.18)

chainE = emcee_trace #[0]
M = np.size(chainE)

ax1 = fig.add_subplot(121)
xgrid = np.linspace(1, M, M)
plt.plot(xgrid, chainE)
ax1.axis([0, M, np.min(chainE), 1.1*np.max(chainE)])
plt.xlabel('number',fontsize=15)
plt.ylabel('chain',fontsize=15)

# plot running mean: 
meanC = [np.mean(chainE[:int(N)]) for N in xgrid]
ax1.plot(xgrid, meanC, c='red', label='chain mean') 
ax1.plot(xgrid, 0*xgrid + np.mean(data),
         c='yellow',label='data mean')
ax1.legend(fontsize=15)

ax2 = fig.add_subplot(122)
# skip first burn samples
Nburn = 1000
Nchain = np.size(chainE[xgrid>burn])
Nhist, bins, patches = plt.hist(chainE[xgrid>Nburn], 
                                bins='auto', histtype='stepfilled')

# plot expectations based on central limit theorem
binwidth = bins[1] - bins[0]
muCLT = np.mean(data)
sigCLT = np.std(data)/np.sqrt(len(data[:,0]))
muGrid = np.linspace(0.7, 1.3, 500)
gauss = Nchain * binwidth * stats.norm(muCLT, sigCLT).pdf(muGrid) 
ax2.plot(muGrid, gauss, c='red') 

ax2.set_ylabel('p(chain)',fontsize=15)
ax2.set_xlabel('chain values',fontsize=15)
ax2.set_xlim(0.7, 1.3)
ax2.set_ylim(0, 1.2*np.max(gauss))
ax2.set_title(r'Chain from emcee',fontsize=15)
