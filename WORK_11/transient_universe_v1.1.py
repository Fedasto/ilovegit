#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:35:00 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams['font.size'] = 12
from scipy import stats
import emcee
import time
import corner

def burst_model(b,A,t0,alpha,t): # myposterior y=b+A*np.exp(-alpha*(t-t0))
    return (np.where(t<t0,b,b+A*np.exp(-alpha*(t-t0))))

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

b_par=10
A_par=5
t0_par=50
alpha_par=0.1

xx=np.linspace(-1,101,100)
plt.plot(xx ,burst_model(b_par,A_par,t0_par,alpha_par,xx), c='red', label=r'Fit params ; b=10 ; A=5 ; $t_0=50$ ; ${\alpha}$ = 0.1 ')
plt.legend(loc='best')

#%%
x,y,sigma_y=data[:,0],data[:,1],data[:,2]
b,A,t0,alpha=np.random.uniform(0,50), np.random.uniform(0,50), np.random.uniform(0,100), np.random.uniform(-5,5)
t0min,t0max = 0,100
Amin,Amax=0,50
bmin,bmax=0,50
alphamin,alphamax=np.exp(-5),np.exp(5)

def Likelihood(theta, data, model=burst_model):
    # Gaussian likelihood 
    
    
    y_fit = burst_model(b,A,t0,alpha, x)
    
    return sum(stats.norm.pdf(*args) for args in zip(y, y_fit, sigma_y))

def LogPrior(x):
    if Amin < A < Amax and bmin < b < bmax and t0min < t0 < t0max and alphamin < alpha < alphamax:
        return 0.0 + 0.0 + 0.0 -np.log(alpha)
    return - np.inf
   

def myPosterior(theta,data,x):
    return Likelihood(theta, data) # * Prior(x)

# emcee wants ln of posterior pdf
def myLogPosterior(theta,data,x):
    return np.log(myPosterior(theta, data, x)) + LogPrior(x)

ndim = 4  # number of parameters in the model
nwalkers = 20  # number of MCMC walkers
burn = 1000  # "burn-in" period to let chains stabilize
nsteps = int(1e4)  # number of MCMC steps to take **for each walker**

# initialize theta 
np.random.seed(0)
starting_guesses = np.random.random((nwalkers, ndim))

theta=b,A,t0,alpha
t_start=time.time()

# the function call where all the work happens: 
sampler = emcee.EnsembleSampler(nwalkers, ndim, myLogPosterior, args=[theta, data])
sampler.run_mcmc(starting_guesses, nsteps)
 
# sampler.chain is of shape (nwalkers, nsteps, ndim)
# throw-out the burn-in points and reshape:
emcee_trace  = sampler.chain[:, burn:, :].reshape(-1, ndim)

t_stop=time.time()
print("done in time=%1.3f \n" % (t_stop-t_start))
print('sampler chain shape \n',sampler.chain.shape) #original chain structure
print('emcee race shape',emcee_trace.shape) #burned and flattened chain

#%%


# plot 
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(30, 20))
plt.title('Trace plots')

labels=['b','A','t0','alpha']
t_grid=np.linspace(0,180000,180000)


for i in range(4):
    
    plt.subplot(2,2,i+1)
    plt.plot(t_grid,emcee_trace[:,i], c='black')
    plt.ylabel(labels[i])
    plt.xlabel('time')


fig, axes = plt.subplots(4, figsize=(30, 20), sharex=True)


samples = sampler.get_chain()

labels = ["A","b","t0","alpha"]
'''for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");'''

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(samples[:,:,i], c='black')
    plt.ylabel(labels[i])
    plt.xlabel('time')
    plt.ylim(0,1)
    
    
#%%

# thinning

tau = sampler.get_autocorr_time()
print('If small increase the MCMC time of performing', tau)

# Since I have data=arviz.InferenceData object I use the ensambleSampler sampler.get_chain
tau = [69.35368153, 62.64704868, 71.69810494, 58.34767926]
flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)
print(flat_samples.shape)


# corner plot


fig=plt.figure(figsize=(20,20),dpi=100)
fig=corner.corner(data, labels=labels,levels=[0.68,0.95] ,quantiles=(0.16, 0.84),show_titles=True,title_fmt='g', use_math_text=True, fig=fig)
fig.show()
