# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 19:09:39 2022

@author: PC
"""

import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import uniform
import scipy.stats
from scipy import optimize
from astroML import stats as astroMLstats
from astropy.visualization.hist import hist as fancyhist
from sklearn.neighbors import KernelDensity
import tqdm
from scipy.integrate import quad

def M_irr(x,y):
    return x*np.sqrt((1+np.sqrt(1-y**2))/(2))

def f(x):
    return M_irr(mass,chi)/x

N=10000

# check f of chi

mass=np.linspace(0,100,N)
xx = np.linspace(0,1,N) # chi
plt.figure(figsize=[5,5])
plt.plot(xx,M_irr(mass,xx)/mass, c='black')
plt.xlabel('$\chi$')
plt.ylabel('f')

# distribution of f

plt.figure()
chi=np.random.uniform(0,1,N)
f=f(mass)
plt.hist(f,density=True, histtype='step',bins=50, color='black')

ff = np.linspace(1/2**0.5,1,100)
pdf_f = 2*(2*ff**2-1)/(1 - ff**2)**0.5
plt.plot(ff,pdf_f,lw=2, color='red')
plt.xlabel('f')

# Distribution of masses

plt.figure()
x_grid= np.linspace(1-5*0.02,1+5*0.02,N)
mass=norm.pdf(x_grid,1,0.02)
plt.plot(x_grid,mass)

MASS=np.random.normal(loc=1,scale=0.02,size=N)
plt.hist(MASS,density=True,histtype='step',bins=80,lw=2);
plt.xlabel('$M$')

# Irriducible mass distribution

plt.figure()
plt.hist(M_irr(MASS,chi)*MASS, density=True, histtype='step', bins=80, color='black')
plt.title('irriducible mass')

# check by numerical integration
scale=0.02
x = np.linspace(1/np.sqrt(2),1+5*scale,N)

def integrand(f,x):
    return ((2/np.pi)**0.5 / scale ) * np.exp(-(x/f -1)**2 /(2*scale**2)) * (2*f**2-1)/(1 - f**2)**0.5 / f


Mirr = [quad(lambda littlef: integrand(littlef,xt), 1/2**0.5,1)[0] for xt in enumerate(x)]

plt.plot(x,Mirr)
plt.xlabel('$M_{\\rm irr}$');


