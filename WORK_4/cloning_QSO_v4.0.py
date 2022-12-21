# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:25:54 2022

@author: PC
"""

import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_dr7_quasar
from scipy.stats import rv_histogram
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 # The astropy.cosmology sub-package contains classes for representing cosmologies and utility functions for calculating commonly used quantities that depend on a cosmological model. This includes distances, ages, and lookback times corresponding to a measured redshift or the transverse separation corresponding to a measured angular separation.




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

qso_pdf=4*np.pi*Planck15.differential_comoving_volume(z).value
plt.figure()
plt.hist(qso_pdf)
plt.title('comoving volume at differen redshift')

counts,bins=np.histogram(qso_pdf, density=True)

dist_qso=rv_histogram([counts,bins])

plt.figure()
x_grid=np.linspace(0,100,10000)
plt.plot(x_grid, dist_qso.pdf(x_grid))
plt.title('histogram distribution')


bin_mids= (bins[1:] + bins[:-1]) / 2
model_dist_qso=interp1d(bin_mids, dist_qso.pdf(bin_mids), fill_value='extrapolate')

plt.figure()
plt.plot(x_grid , model_dist_qso(x_grid))
plt.title('interpolated pdf')

plt.figure()
plt.hist(z, bins=100, Fill=False, density=True,label='dataset QSOs redshift')
plt.plot(x_grid , model_dist_qso(x_grid)*1e9, color='red', label='fit model')
plt.legend(loc='best')


