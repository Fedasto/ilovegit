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
from scipy.interpolate import interp1d
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

''' nope
plt.figure()
plt.plot(dist_histo.cdf(x_grid), x_grid, color='orange')
plt.title('inverse cdf')'''

N_sampling=1000

bin_mids = (bins[1:] + bins[:-1]) / 2 # mid location of bins
xx=np.linspace(0,6,100)
 
plt.figure()
inv_dist_histo = interp1d(dist_histo.cdf(bin_mids), bin_mids, fill_value='extrapolate')
plt.title('interpolated inverted cdf ')
plt.plot(xx, inv_dist_histo(xx), color='orange')

u=np.random.uniform(0,1,N_sampling)
sampling=inv_dist_histo(u)

plt.figure()
plt.hist(sampling, bins=100, density=True, label='cloned', fill=False, color='red')
plt.hist(z, bins=100, density=True, label='real', alpha=0.3)
plt.title('inverse transform cloning method')
plt.legend(loc='best')