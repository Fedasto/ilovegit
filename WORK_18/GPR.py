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
c=3*10**8
#omega_m=0.31
omega_l=0.69

integranda = lambda x1,omega_m: 1/(omega_m*(1+x1**3) + omega_l)

mu = lambda omega_m,H0,x: 5*np.log((1+x)*(c/H0)/(10))*list(quad(integranda,0,x, args=(omega_m))[0])


popt, pcov = curve_fit(mu, X[:,0], y, p0=[0.31,70], method='lm')

plt.plot(X, mu(X, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))






