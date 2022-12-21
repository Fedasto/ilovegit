# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:59:52 2022

@author: PC
"""

# Inverse transform sampling method

import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.stats import rv_histogram, norm
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

plt.rcParams['figure.figsize'] = [8, 8]

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

z = data['redshift']
#%%
plt.hist(z, fill=False, bins=100, density=True, label='real redshift data')
plt.legend()

count,bins=np.histogram(z,bins=100,density=True)

dist_histo=rv_histogram([count,bins])

x_grid=np.linspace(0,6,10000)

plt.figure()
plt.hist(count,bins, color='green', fill=False)
plt.title('input data distribution')

plt.figure()
plt.plot(x_grid, dist_histo.cdf(x_grid), color='red')
plt.title('Cumulative distribution')

plt.figure()
plt.plot(dist_histo.cdf(x_grid), x_grid, color='orange')
plt.title('inverse cdf')

N_sampling=10000
sampling=[]

plt.figure()
inv_dist_histo = np.quantile(dist_histo.cdf, q=0.5, method='inverted_cdf')
plt.title('inverted cdf is the quantile function')
plt.plot(x_grid, inv_dist_histo(x_grid), color='pink')

for i in range(N_sampling):
    sampling.append(inv_dist_histo.rvs())

plt.figure()
plt.hist(sampling, bins=100, density=True)
plt.hist(count, bins=bins, density=True)
plt.legend(loc='best')



