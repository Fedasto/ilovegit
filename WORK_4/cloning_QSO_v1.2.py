# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:58:25 2022

@author: PC
"""

import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.stats import poisson
# %matplotlib inline
# %config InlineBackend.figure_format='retina'


plt.rcParams['figure.figsize'] = [8, 8]

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

z = data['redshift']

# Let me plot the redshift histogram
x_grid=np.arange(0, np.max(z), 1)
mu=np.arange(0.0, 3, 0.5)

# try to fit the histogram with a poisson pdf, searching for the best mu
plt.figure()
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')
for i in range(len(mu)):
    plt.plot(x_grid, poisson(mu[i]).pmf(x_grid), label=r'poisson with $\mu=%1.1f$' % (mu[i]))
plt.legend(loc='best')

# Define the sampling coordinate
N_sampling=10000

x_sampling=np.random.uniform(0, 10, N_sampling)
y_sampling=np.random.uniform(0, np.max(z), N_sampling)

x_grid=np.linspace(0, 10, len(z))

plt.figure()
plt.scatter(x_sampling, y_sampling, s=10, c='red', marker='o', alpha=1)
plt.scatter(x_grid, z, s=10, c='blue', marker='o', alpha=0.2)

#%%
# define the possion function 
poisson=lambda x: (poisson(1.5).pmf(x)) 
#find the point whose satisfies the rejection sampling condition
good_pts=x_sampling[z<=poisson(x_sampling)] #x[y<f(x)]

# plot the results
plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')


plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')

plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')
plt.plot(x_grid, poisson(x_grid))