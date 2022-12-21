# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:46:28 2022

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

qso_pdf=list(4*np.pi*Planck15.differential_comoving_volume(z).value)
x_grid=np.linspace(np.min(qso_pdf),np.max(qso_pdf),10000)
plt.figure()
plt.hist(qso_pdf, density=True, bins=100, fill=False)
plt.hist(qso_pdf)
plt.title('comoving volume at differen redshift')

counts, bins=np.histogram(qso_pdf, density=True)

dist_qso_pdf = interp1d((bins[1:]-bins[:-1] / 2) , counts, fill_value='extrapolate')

xx=np.linspace(1e10,5e11,10000)
plt.plot(xx, dist_qso_pdf(xx)*1e14)

xg=np.linspace(1e10,5e11,100)
plt.figure()
plt.hist(z*1e11,bins=100, fill=False, density=True)
plt.plot(xg, dist_qso_pdf(xg))





