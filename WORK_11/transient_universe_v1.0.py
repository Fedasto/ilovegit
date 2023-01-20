#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:31:09 2023

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
plt.plot(xx ,burst_model(b_par,A_par,t0_par,alpha_par,xx), c='red', label=r'Fit params ; b=10 ; A=10 ; $t_0=50$ ; ${\alpha}$ = 0.3 ')
plt.legend(loc='best')

#%%
x,y,sigma_y=data[:,0],data[:,1],data[:,2]
b,A,t0,alpha=np.random.uniform(0,50), np.random.uniform(0,50), np.random.uniform(0,100), np.random.uniform(-5,5)
t0min,t0max = 0,100
Amin,Amax=0,50
bmin,bmax=0,50
alphamin,alphamax=-5,5

def Likelihood(theta, data, model=burst_model):
    # Gaussian likelihood 
    
    
    y_fit = burst_model(b,A,t0,alpha, x)
    
    return sum(stats.norm.pdf(*args) for args in zip(y, y_fit, sigma_y))

def Prior(x):
    if Amin < A < Amax and bmin < b < bmax and t0min < t0 < t0max and alphamin < alpha < alphamax:
        return 0.0 + 0.0 + 0.0 + 1/(alpha)
    return 1/np.inf
   

def myPosterior(theta,data,x):
    return Likelihood(theta, data) * Prior(x)

# emcee wants ln of posterior pdf
def myLogPosterior(theta,data,x):
    return np.log(myPosterior(theta, data, x))

ndim = 4  # number of parameters in the model
nwalkers = 10  # number of MCMC walkers
burn = 1000  # "burn-in" period to let chains stabilize
nsteps = 5000  # number of MCMC steps to take **for each walker**

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
print("done in time=%1.3f " % (t_stop-t_start))
print('sampler chain shape \n',sampler.chain.shape) #original chain structure
print('emcee race shape',emcee_trace.shape) #burned and flattened chain

#%%


# plot 
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize=(15, 8))
#fig.subplots_adjust(left=0.11, right=0.95, 
                   # wspace=0.35, bottom=0.18)

chainE = emcee_trace.flatten() #[0]
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
#%%

# thinning

tau = sampler.get_autocorr_time()
print('If small increase the MCMC time of performing', tau)

# Since I have data=arviz.InferenceData object I use the ensambleSampler sampler.get_chain

flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)
print(flat_samples.shape)


# corner plot

labels = [str(theta)]
fig=plt.figure(figsize=(7,7),dpi=100)
fig=corner.corner(data, labels=labels, quantiles=(0.16, 0.84),show_titles=True,title_fmt='g', use_math_text=True, fig=fig)
fig.show()

#%%
sub_sample = np.random(flat_samples,100)

plt.figure(figsize=(12,8))
plt.plot(xx,sub_sample, alpaha=0.5, color='gray', label='mcmc sampling result')
plt.scatter(data[:,0],data[:,1], marker='o',color='black')
plt.errorbar(data[:,0],data[:,1],yerr=data[:,2], fmt='o', color='black', capsize=2)
plt.xlabel('time')
plt.ylabel('flux')
plt.title('Transient')
plt.legend(loc='best')

#%%

for i,l in enumerate(labels):
    low,med, up = np.percentile(flat_samples[:,i],[5,50,95]) 
    print(l+"\t=\t"+str(round(med,2))+"\t+"+str(round(up-med,2))+"\t-"+str(round(med-low,2)))




