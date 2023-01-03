# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 12:38:25 2022

@author: PC
"""



import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format='retina'

import scipy.stats as stats
from astroML import stats as astroMLstats
from astropy.visualization.hist import hist as fancyhist
from sklearn.neighbors import KernelDensity

N=10000
chi = np.random.uniform(0,1,N)
mass = np.random.normal(loc=1, scale=0.002,size=N) # mean is a scale not a number, all is in units of this


def M_irr(x,y):
    return x*np.sqrt((1+np.sqrt(1-y**2))/(2))

def f(x):
    return M_irr(mass,chi)/x

f=f(mass)
'''
plt.figure()
plt.plot(chi,f)
plt.xlabel('$\chi$')
plt.ylabel('$f$')

plt.figure()
plt.hist(np.random.normal(loc=1,scale=0.02,size=N),bins=100, histtype='step', density=True)
plt.plot(np.linspace(1-5*0.02,1+5*0.02,N), mass)

'''

'''
plt.figure(figsize=[8,8])
plt.title('$\frac{M_{irr}{M}$ histogrma not knowing bin size')
bins=[10,50,100]
for i in range(len(bins)):
    plt.hist(f,bins=bins[i], label='bins = %1.0f' % i)
plt.legend(loc='best')'''

bin_size = 2.7*astroMLstats.sigmaG(f) / N**(1/3) # Friedman - Diaconis rule also if we know sigma
print('bins size from F-D rule is ', bin_size)

plt.figure(figsize=[8,8])
plt.title('Histogram with the correct bin size')
_ = fancyhist(f, bins="scott", histtype="step",density=True, label='scott')
_ = fancyhist(f, bins="freedman", histtype="step",density=True, label='freedman') # ‘freedman’ : use the Freedman-Diaconis rule to determine bins
plt.legend(loc='best')

plt.figure(figsize=[8,8])
plt.title('rug plot')
plt.hist(f,histtype="step")
plt.plot(f[:100], 0*f[:100], '|', color='k', markersize=25) #Note markersize is (annoyingly) in *points*

plt.figure(figsize=[5,5])
xgrid = np.linspace(f.min(),f.max(),1000)

def kde_sklearn(data, bandwidth = 1.0, kernel="linear"):
    kde_skl = KernelDensity(bandwidth = bandwidth, 
                            kernel=kernel)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)

    return np.exp(log_pdf)

plt.title('KDE plot of $M_{irr}$')

PDFtophat = kde_sklearn(f,bandwidth=0.01,kernel="gaussian") #Complete
plt.plot(xgrid,PDFtophat, label='tophat + gaussian') #Complete

PDFtophat = kde_sklearn(f,bandwidth=0.02,kernel="epanechnikov") #Complete
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
for j in range(len(mass)):
    print('sigma =', sigma[j])
    ks_test_m_f.append(stats.kstest(mass[j],f[j]))
    print(stats.kstest(mass[j],f[j]))
    ks_test_m_mirr.append(stats.kstest(mass[j], M_irr(mass[j],chi)))
    print(stats.kstest(mass[j], M_irr(mass[j],chi)))

plt.figure(figsize=[8,8])
for w in range(len(ks_test_m_f)):    
    plt.scatter(ks_test_m_f[w][0], ks_test_m_f[w][1], label='sigma = %1.0f ' % sigma[w])    
    plt.title('KS test mass and f')
    plt.xlabel('statistic')
    plt.ylabel('pvalue')
    
'''    
plt.figure(figsize=[8,8])
for w in range(len(ks_test_m_f)):    
    plt.hist(ks_test_m_f[w][0], label='sigma = %1.0f ' % sigma[w])    
    plt.title('KS test mass and f')
    plt.xlabel('statistic')
    
plt.figure(figsize=[8,8])
for w in range(len(ks_test_m_f)):    
    plt.hist(ks_test_m_f[w][1], label='sigma = %1.0f ' % sigma[w])    
    plt.title('KS test mass and f')
    plt.xlabel('pvalue')
'''    
    
plt.figure(figsize=[8,8])
for w in range(len(ks_test_m_mirr)):    
    plt.scatter(ks_test_m_mirr[w][0], ks_test_m_mirr[w][1], label='sigma = %1.0f ' % sigma[w])    
    plt.title('KS test mass and $M_{irr}$')
    plt.xlabel('statistic')
    plt.ylabel('pvalue')    
    



