#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 18:43:18 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
import itertools
from scipy.integrate import quad
import emcee
import scipy.stats as scistats
import corner
import dynesty
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib.colors import ListedColormap
#%%





z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)
z = np.linspace(0.01, 2, 100)

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)


# Gaussian Process regression GPR

X=z[:,None]
gpr= GaussianProcessRegressor(kernel=(ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))), alpha=dmu ** 2)
gpr.fit(z_sample[:,None],mu_sample)

y_pred, std_y_pred = gpr.predict(X, return_std=True)

y_pred = gpr.predict(z[:,np.newaxis])

plt.plot(z,y_pred,'r--', label='GPR fit')
plt.fill_between(z, y_pred - std_y_pred, y_pred + std_y_pred, color="orange", alpha=0.2, label=r'$1\sigma$ level')
plt.fill_between(z, y_pred - 2*std_y_pred, y_pred + 2*std_y_pred, color="yellow", alpha=0.2, label=r'$2\sigma$ level')
plt.legend(loc=0)
plt.show()


#%%

# cross validation for kernel hyperparameter 

X=z_sample[:,np.newaxis]
y=mu_sample

def cross_val_kernel(kernels, X, Y):
    performance = {}
    
    for kernel in tqdm(kernels): 
        likelihood = 0
        for i in range(Y.size):
            gp = GaussianProcessRegressor(kernel = kernel)
            X_train = np.delete(X, i, axis=0)
            Y_train = np.delete(Y, i, axis=0)
            gp.fit(X_train, Y_train)
            y_mean, y_std = gp.predict(X[[i], :], return_std = True)
            likelihood += -np.log(y_std[0]) - (Y[i] - y_mean[0])**2 / (2 * y_std[0]**2)
            
        performance[likelihood] = kernel
        
    return performance

parRange = [np.arange(0.5, 2, 0.2), np.arange(5, 15, 1)] # , np.arange(0.1, 1, 0.1)
kernels = [ConstantKernel(a) * RBF(b) for a, b in list(itertools.product(*parRange))] #  WhiteKernel(c)

performance = cross_val_kernel(kernels, X, y)

best_key = np.max(list(performance.keys()))

print('best kernel hyperparameters = ', performance[best_key])





#%%

# PART 2 MCMC parameter estimation for complex model (lambda CDM)


#H0=70
c=3*10**5 #km/sec
#omega_m=0.31
omega_l=0.69

def func(x,omega_m,H0):
    fn = lambda t: 1/(np.sqrt(omega_m*(1+t**3)+omega_l))
    integral = np.asarray([quad(fn, 0, _x)[0] for _x in x])
    mu = 5*np.log((c/H0)/1 * (1+x) * list(integral))
    return mu

# gaussian likelihood

def LogLikelihood(theta):
    Om,H0 = theta    
    if Om<0:
        return -np.inf
    else:
        mu_model = func(z_sample,Om,H0)
    
    #return np.sum(scistats.norm.logpdf(mu_sample, loc=mu_model, scale=dmu))
    return np.sum(scistats.norm(loc=mu_model, scale=dmu).logpdf(mu_sample))

def Logprior(theta):
    Om,H0 = theta
    if 50 < H0 < 100 and 0.1 < Om < 1:
        return 0.0
    return -np.inf

                   
def LogPosterior(theta):
    return LogLikelihood(theta) + Logprior(theta)


ndim = 2  # number of parameters in the model
nwalkers = 5  # number of MCMC walkers
nsteps = 10000  # number of MCMC steps to take **for each walker**

starting_guesses = np.array([0.5,80]+1e-1*np.random.random((nwalkers, ndim)))
start=time.time()

sampler = emcee.EnsembleSampler(nwalkers, ndim, LogPosterior)
sampler.run_mcmc(starting_guesses, nsteps)

stop=time.time()
print("done in %1.3f" % (stop-start))

samples = sampler.get_chain()
tau = sampler.get_autocorr_time()
flat_samples = sampler.get_chain(discard=3*int(max(tau)), thin=int(max(tau)), flat=True)


# corner plot

fig = corner.corner(flat_samples, labels=[r"$\Omega_m$","$H_0$"], levels=[0.68,0.95], show_titles=True, truths=[0.27,71], quantiles=[0.16,0.84])



# plot the MCMC par result

zz=np.linspace(0,2,100)[1:]

plt.figure()
for Om,H0 in flat_samples[::100]:
    plt.plot(zz,func(zz,Om,H0), 'r-', alpha=0.2)

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)

#%%

# Part 3

# PCA to look at last 2 points

X_train, X_test, y_train, y_test, dy_train, dy_test = train_test_split(X, y, dmu, test_size = 0.2, random_state = 0)

sc = StandardScaler()
  
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pca = PCA(n_components = 1)
  
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
  
explained_variance = pca.explained_variance_ratio_

classifier = GaussianProcessRegressor(kernel=(ConstantKernel(105) * RBF(14)), alpha=dy_train ** 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

plt.figure()
plt.scatter(X_train,y_train,c=dy_train, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))

# Add the color bar
plt.colorbar(label='uncertianities weight', ticks=[0,1])

# Make it clear which color goes with which label
plt.clim(np.min(dmu),np.max(dmu))





#%%
# Part 4

# 10times larger dataset

def func(x,omega_m,H0):
    fn = lambda t: 1/(np.sqrt(omega_m*(1+t**3)+omega_l))
    integral = np.asarray(quad(fn, 0, x)[0])
    mu = 5*np.log((c/H0)/1 * (1+x) * (integral))
    return mu

z_vals = np.random.uniform(0,2,1000)
mu_mcmc = []
mu_gpr=[]

for i in range(len(z_vals)):
    omega_matter,H_0 = np.random.choice(flat_samples[:,0]), np.random.choice(flat_samples[:,1])
    mu_model=func(z_vals[i],omega_matter,H_0)
    mu_mcmc.append(mu_model)

mu_pred, dmu = gpr.predict(z_vals[:,np.newaxis], return_std=True)
mu_gpr.append(np.random.normal(loc=mu_pred, scale=dmu))

plt.figure()    
plt.scatter(z_vals,mu_mcmc,alpha=0.2,label="emcee sampling")
plt.scatter(z_vals,mu_gpr[0],alpha=0.2,label='GPR')

plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)
plt.title("Cloned data")
plt.legend(loc=0)   

#%%

# Part 5 

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

# What if we avoid Dark Energy (Om==1). Let's use Nasted sampling in this case

#H0=70
c=3*10**5 #km/sec
#omega_m=0.31
omega_l=0.69

def func(x,omega_m,H0):
    fn = lambda t: 1/(np.sqrt(omega_m*(1+t**3)+omega_l))
    integral = np.asarray([quad(fn, 0, _x)[0] for _x in x])
    mu = 5*np.log((c/H0)/1 * (1+x) * list(integral))
    return mu

def LogLikelihood_1(theta):
    Om,H0 = theta    
    Om=1
    mu_model = func(z_sample,Om,H0)
    
    #return np.sum(scistats.norm.logpdf(mu_sample, loc=mu_model, scale=dmu))
    return np.sum(scistats.norm(loc=mu_model, scale=dmu).logpdf(mu_sample))

def prior_transform(u):
    mins = np.array([30.])
    maxs = np.array([100.])
    
    return  mins + u*(maxs-mins)  # uniform distribution 

sampler = dynesty.NestedSampler(LogLikelihood_1, prior_transform, ndim=2, nlive=500)
sampler.run_nested()
sresults = sampler.results

rfig, raxes = dyplot.runplot(sresults)
tfig, taxes = dyplot.traceplot(sresults)



samples = sresults.samples_equal()
weights = np.exp(sresults.logwt - sresults.logz[-1])
samples_equal = dyfunc.resample_equal(samples, weights)

#plt.hist(samples_equal, bins=100)

fig = corner.corner(samples_equal, labels=["Om", "H0"], levels=[0.68,0.95], show_titles=True, truths=[1,71], quantiles=[0.16,0.84])

# Compute 10%-90% quantiles.
quantiles = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights)
             for samps in samples.T]
print('68% parameter credible regions are:\n ' + str(quantiles) + '\n')

# Compute weighted mean and covariance.
mean, cov = dyfunc.mean_and_cov(samples, weights)
print('Mean and covariance of parameters are: ' + str(mean) + '\n' + str(cov))




