# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:12:30 2022

@author: PC
"""

import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import scipy.stats
from scipy.stats import norm, poisson, t
from scipy.stats import uniform
from astroML import stats as astroMLstats
from astroML.datasets import fetch_dr7_quasar


plt.rcParams['figure.figsize'] = [8, 8]

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

z = data['redshift']
#%% t_student fit function
x_grid=np.arange(0,np.max(z),1)
df=np.arange(4,5,0.1)

plt.figure()
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')
for i in range(len(df)):
    plt.plot(x_grid, t.pdf(x_grid, df=df[i]), label=r't-student with df=%1.1f' % (df[i]))
plt.legend(loc='best')



#%%
N_sampling=10000

x_sampling=np.random.uniform(0,10,N_sampling)
y_sampling=np.random.uniform(0,np.max(z),N_sampling)

x_grid=np.linspace(0,10,len(z))

plt.figure()
plt.scatter(x_sampling,y_sampling, s=10, c='red', marker='o', alpha=1)
plt.scatter(x_grid,z, s=10, c='blue', marker='o', alpha=0.2)

#%%
normal= lambda x: (norm(loc=np.mean(z),scale=np.std(z)).pdf(x)) 
good_pts=x_sampling[z<=normal(x_sampling)] #x[y<f(x)]

plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')

#%%
plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')
plt.hist(z, fill=False, bins=100, density=True, label='redshift data')

plt.figure()
plt.hist(good_pts, histtype='step', density=True, color='green')
plt.plot(x_grid, normal(x_grid))