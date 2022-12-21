# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:37:53 2022

@author: PC
"""

import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.stats import rv_histogram
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 # The astropy.cosmology sub-package contains classes for representing cosmologies and utility functions for calculating commonly used quantities that depend on a cosmological model. This includes distances, ages, and lookback times corresponding to a measured redshift or the transverse separation corresponding to a measured angular separation.
from scipy.integrate import quad



#%matplotlib inline
#%config InlineBackend.figure_format='retina'

plt.rcParams['figure.figsize'] = [8, 8]

# Fetch the quasar data
data = fetch_dr7_quasar()

# select the first 10000 points
data = data[:10000]

z = data['redshift']

#%%
#How do QSOs are distributed in space?

qso_pdf = lambda z_data: 4*np.pi*Planck15.differential_comoving_volume(z_data).value

def normalization (func):
    return quad(qso_pdf, 0, 5)[0]

norm_dist = (qso_pdf(z) / normalization(qso_pdf(z)))


plt.figure()
plt.hist(z, bins=100, histtype='step', label='redshift dataset', density=True)
plt.plot(np.sort(z), 2.5*norm_dist, label='model trend')
plt.legend(loc='best')






