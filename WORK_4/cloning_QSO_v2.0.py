# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 14:50:28 2022

@author: PC
"""

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
x_grid=np.arange(0,np.max(z),1)
mu=np.arange(0.0,3,0.5)

plt.figure()
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')
plt.legend(loc='best')
counts, bins = np.histogram(z, bins=100, density=True)

dist_histo=rv_histogram([counts,bins])


#%%
N_sampling=10000

x_sampling=np.random.uniform(0,10,N_sampling)
y_sampling=np.random.uniform(0,np.max(z),N_sampling)

x_grid=np.linspace(0,10,len(z))

plt.figure()
plt.scatter(x_sampling,y_sampling, s=10, c='red', marker='o', alpha=1, label='sampling')
plt.scatter(x_grid,z, s=10, c='blue', marker='o', alpha=0.2, label='real')
plt.title('sampling data coordinates')
plt.legend(loc='best')

#%% 
good_pts=x_sampling[z<=dist_histo.pdf(x_sampling)] #x[y<f(x)]

plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')

#%%
plt.figure()
plt.title('Montecarlo rejection method')
plt.hist(good_pts, histtype='step', density=True, color='green', label='cloned data')
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')
plt.legend(loc='best')

plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green', label='cloned histogram')
plt.plot(x_grid,dist_histo.pdf(x_grid), label='real histogram')
plt.legend(loc='best')


#%%
normal= lambda x: (norm(loc=np.mean(z),scale=np.std(z)).pdf(x)) 
good_pts_norm=x_sampling[z<=normal(x_sampling)] #x[y<f(x)]

plt.figure()
plt.hist(good_pts_norm, histtype='step', density=True, color='orange')

#%%
plt.figure()
plt.hist(good_pts_norm, histtype='step', density=True, color='orange')
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')

plt.figure()
plt.hist(good_pts_norm, histtype='step', density=True, color='orange')
plt.plot(x_grid, normal(x_grid))

# comparison between gaussian and rv_histogram

plt.figure()
plt.hist(good_pts_norm, histtype='step', density=True, color='orange', label='gaussian sample')
plt.hist(z, fill=False, bins=100, density=True, label='real redshift data')
plt.hist(good_pts, histtype='step', density=True, color='green', label='rv histogram sample')
plt.legend(loc='best')