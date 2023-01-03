# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 17:00:22 2022

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

N=10000
chi = np.random.uniform(0,1,N)
mass = np.random.normal(loc=1, scale=0.02,size=N) # mean is a scale not a number, all is in units of this

# check the mass distribution is gaussian
plt.figure()
x_grid=np.linspace(np.min(mass),np.max(mass),N)
plt.hist(mass,bins=100,density=True, histtype='step', color='black')
plt.plot(x_grid, norm(loc=1,scale=0.02).pdf(x_grid), '-', c='red')
plt.title('The masses follow a gaussian pdf')

# check the chi distribution is uniform
plt.figure()
# xg = np.linspace(0,1,N)
plt.hist(chi,color='black', fill=False)
plt.title('uniform distribution of $\chi$')



def M_irr(x,y):
    return x*np.sqrt((1+np.sqrt(1-y**2))/(2))

def f(x):
    return M_irr(mass,chi)/x

f=f(mass)

# check the f(chi) ditribution

xx = np.linspace(0,1,N)
plt.figure(figsize=[5,5])
plt.plot(xx,M_irr(mass,xx)/mass, c='black')
plt.xlabel('$\chi$')
plt.ylabel('f')


plt.figure(figsize=[5,5])
plt.title(r'$\frac{M_{irr}}{M}$ histogrma not knowing bin size')

bins=[20,80,100]
ff = np.linspace(1/2**0.5,1,N)
pdf_f = 2*(2*ff**2-1)/(1 - ff**2)**0.5

# histo with different binnings fitted by the PDF of f (=M_irr formula)

for i in range(len(bins)):
    plt.hist(f,bins=bins[i], density=True, histtype='step',lw=2, label='bins = %1.0f' % bins[i])
plt.plot(ff,pdf_f,'--',label='pdf of f')
plt.xlabel('f')
plt.ylim(0,55)
plt.legend(loc='best')

bin_size = 2.7*astroMLstats.sigmaG(f) / N**(1/3) # Friedman - Diaconis rule also if we know sigma
print('bins size from F-D rule is ', bin_size)

# plot the irriducible mass without normalization to m

xx=np.linspace(1-5*0.02,1+5*0.02,N)

plt.figure(figsize=[8,8])
plt.title('Histogram with the correct bin size')
_ = fancyhist(M_irr(mass,chi), bins="scott", histtype="step",density=True, label='scott') # f*norm.pdf(xx,1,0.02)
_ = fancyhist(M_irr(mass,chi), bins="freedman", histtype="step",density=True, label='freedman') # ‘freedman’ : use the Freedman-Diaconis rule to determine bins
#plt.hist(f,bins=[bin_size for i in range(len(f))], histtype='step', label='home made')
plt.legend(loc=6)

plt.figure(figsize=[8,8])
plt.title('rug plot')
plt.hist(M_irr(mass,chi),histtype="step", bins=80, label='irriducible mass')
plt.plot(M_irr(mass,chi)[:100], 0*M_irr(mass,chi)[:100], '|', color='k', markersize=25) #Note markersize is (annoyingly) in *points*
plt.legend(loc=0)

#%%

# Part 2 - Limits





plt.figure(figsize=[5,5])
xgrid = np.linspace(f.min(),f.max(),1000)



def kde_sklearn(data, bandwidth = 1.0, kernel="linear"):
    kde_skl = KernelDensity(bandwidth = bandwidth, 
                            kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)

    return np.exp(log_pdf)

plt.title('KDE plot of $M_{irr}$')

PDFtophat = kde_sklearn(f,bandwidth=0.08,kernel="gaussian") # How do I set the best bandwith
plt.plot(xgrid,PDFtophat, label='tophat + gaussian') #Complete

PDFtophat = kde_sklearn(f,bandwidth=0.08,kernel="epanechnikov") #Complete
plt.plot(xgrid,PDFtophat, label='tophat + epanechnikov') #Complete
plt.legend(loc='best')

#%%

# Part 2
sigma=np.arange(1e-4,1e4,10)
mass=[]
chi = np.random.uniform(0,1,N)
for i in sigma:
    mass.append(np.random.normal(loc=1, scale=i,size=N)) # mean is a scale not a number, all is in units of this



def M_irr(x,y):
    return x*np.sqrt((1+np.sqrt(1-y**2))/(2))

def f(x):
    return M_irr(mass,chi)/x

f=f(mass)

ks_test_m_f=[]
ks_test_m_mirr=[]
for j in range(len(sigma)):
    print('sigma =', sigma[j])
    ks_test_m_f.append(stats.kstest(mass[j],f[j]))
    print(stats.kstest(mass[j],f[j]))
    ks_test_m_mirr.append(stats.kstest(mass[j], M_irr(mass[j],chi)))
    print(stats.kstest(mass[j], M_irr(mass[j],chi)))

x_grid=np.logspace(-4,4,N)

plt.figure(figsize=[8,8])
for w in range(len(sigma)):
    plt.scatter(sigma[w],ks_test_m_f[w][0], label="KS$(M_{\\rm irr}, f)$")
    plt.scatter(sigma[w],ks_test_m_mirr[w][0], label="KS$(M_{\\rm irr}, M)$")
plt.xscale('log')
plt.xlabel('$\sigma$')
plt.ylabel('KS statistics')
plt.legend(loc='best')