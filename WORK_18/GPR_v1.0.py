# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:23:23 2023

@author: PC
"""
import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.integrate import quad
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.stats import chisquare

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)

#%%
X=z_sample
y=mu_sample
dy=dmu


#%%

# choose the kernel here some examples

long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)

seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)

irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)

lincomb_kernel = (long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel)

#%%

# GPR model fitting and extrapolation

X=X[:,np.newaxis]
y_mean = y.mean()
gaussian_process = GaussianProcessRegressor(kernel=1.0*RBF(length_scale=1.0), normalize_y=False)
gaussian_process.fit(X, y - y_mean)

X_test = np.linspace(start=X[0], stop=X[30], ).reshape(-1, 1)
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
mean_y_pred += y_mean

# plot the results
plt.figure()

z=np.linspace(0,2,100)
zz=np.linspace(0,2,50)
plt.plot(z, y, color="black", linestyle="dashed", label="Measurements")
plt.plot(zz, mean_y_pred, color="tab:green", alpha=0.4, label="Gaussian process")
plt.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2, label=r'$1\sigma$')

plt.fill_between(
    X_test.ravel(),
    mean_y_pred - 2*std_y_pred,
    mean_y_pred + 2*std_y_pred,
    color="tab:red",
    alpha=0.2, label=r'$2\sigma$')

plt.legend()
plt.xlabel("z")
plt.ylabel("$<\mu>$")
_ = plt.title(
    "Avarage distances error"
)

# interpretation of hyperparameters

print(gaussian_process.kernel_)


#%%
'''
from astroML.linear_model import NadarayaWatson

plt.figure()
z = np.linspace(0.01, 2, 1000)
regressor = NadarayaWatson('gaussian', 0.04)

regressor.fit(z_sample[:,np.newaxis], mu_sample) # in this case dy is irrelevant!
mu_fit = regressor.predict(z[:,np.newaxis])
   
mu_fit = regressor.predict(z[:,np.newaxis])
plt.plot(z, mu_fit,label='Kernel Regression bw='+str(0.04))

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)'''

#%%

# Part 2 curve_fit = Use non-linear least squares to fit a function, f, to data.

#H0=70
c=3*10**5
#omega_m=0.31
omega_l=0.69

def func(x,omega_m,H0):
    fn = lambda t: 1/(np.sqrt(omega_m*(1+t**3)+omega_l))
    integral = np.asarray([quad(fn, 0, _x)[0] for _x in x])
    mu = 5*np.log((c/H0)/0.01 * (1+x) * list(integral))
    return mu

popt, pcov = curve_fit(func,X[:,0],y, p0=[0.31,68], maxfev=5000)

xx=np.linspace(0,2,100)
plt.figure()
plt.plot(xx, func(X[:,0], *popt) , 'r-',label='fit: $\Omega_m$=%1.3f, $H_0$=%1.3f' % tuple(popt))
plt.title('Best fit')
plt.xlabel('z')
plt.ylabel('$\mu$')
plt.legend(loc=0)

#%%
omega_matter=[]
H_0=[]
Nc=100
limit=[0.31,68]
for i in tqdm(range(Nc)):
    popt, pcov = curve_fit(func,X[:,0],y, p0=limit, maxfev=5000, method='lm')
    
    chisquare = chisquare(y,func(X[:,0]), *popt,ddof=3)
    
    omega_matter_err,H_0_err = np.sqrt(np.diag(pcov)/chisquare)
    
    omega_matter.append([popt[0],omega_matter_err])
    H_0.append([popt[1], H_0_err])

omega_matter=np.array(omega_matter)
H_0=np.array(H_0)

plt.figure()    # These parameters should follow a gaussian pdf
plt.bar(X[:,0],omega_matter[:,0] , color='blue',yerr=omega_matter[:,1])
plt.bar(X[:,0],H_0[:,0],color='red', yerr=H_0[:,1])

plt.hist(omega_matter[:,0], bins=100)
plt.hist(H_0[:,0], bins=100)

# iteratng the integrlas we can determine the best fits parameters using MC mehod in this frequentist approach

    
    
    
    








