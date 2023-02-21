#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:41:05 2023

@author: federicoastori
"""
import numpy as np
import matplotlib.pyplot as plt
from astroML.datasets import generate_mu_z
from sklearn.linear_model import LinearRegression
from astroML.linear_model import PolynomialRegression
from sklearn.model_selection import train_test_split
from astroML.linear_model import BasisFunctionRegression
from astroML.linear_model import NadarayaWatson
from astroML import linear_model

z_sample, mu_sample, dmu = generate_mu_z(100, random_state=1234)

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)

#%% 

X=np.array(z_sample).reshape(100,-1)
y=mu_sample

#%%

# Linear regression

X_new = np.array([[0], [1.75]]) #like X_grid, but just with the endpoints

lin_reg = LinearRegression()
lin_reg.fit(X, y, sample_weight=1.0)

theta0 = lin_reg.intercept_
theta1 = lin_reg.coef_

print(theta0, theta1)
y_pred = lin_reg.predict(X_new)

plt.figure(figsize=(8,5))
plt.plot(X_new,y_pred, 'r--', label='linear regression fit')
plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")
plt.legend(loc='lower right')
plt.xlim(0,2)
plt.ylim(35,50)
#%%

# Polynomial Regression
yy=[]
xx=[]
BIC=[]

for i in range(10):
    X_new=np.linspace(0,1.75,i+2).reshape(i+2,-1)
    xx.append(X_new)
    degree = 2+i
    model = PolynomialRegression(degree) # fit 3rd degree polynomial
    model.fit(X, y)

    y_pred1 = model.predict(X_new)
    yy.append(y_pred1)
    n_constraints = degree + 1
    
    print(model.coef_)

plt.figure(figsize=(15,10))

for j in range(len(xx)):

    plt.plot(xx[j],yy[j], label='polynomial deg = %1.0f' % (j+2))

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")    
plt.legend(loc='lower right', fontsize='x-small')
plt.xlim(0,2)
plt.ylim(35,50)

#%%

# Basis Function Regression

# data
Xtest = X

ytest = y 

dytest=dmu

n_gauss=np.arange(1,20)

plt.figure(figsize=(10,8))

for i in range(2,len(n_gauss),4):
    # mean positions of the 10 Gaussians in the model
    X_gridtest = np.linspace(0,2,i)[:, None]
    # widths of these Gaussians
    sigma_test = 1.0 * (X_gridtest[1] - X_gridtest[0])
    
    model = BasisFunctionRegression('gaussian', mu=X_gridtest, sigma=sigma_test)
    model.fit(Xtest, ytest, dytest)
    
    zz=np.linspace(0.01, 2, 100)
    
    y_pred2 = model.predict(Xtest)
    print(model.coef_)
    
    
    plt.plot(zz,y_pred2,linewidth=1, label='basis regression # gaussian = %1.0f' % (i))

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")    
plt.legend(loc='lower right', fontsize='x-small')
plt.xlim(0,2)
plt.ylim(35,50)

#%%

# Kernel regression

Xtest=X
ytest=y

plt.figure(figsize=(10,8))

height=np.linspace(0.01,0.2,8)

for i in height: 
    
    model = NadarayaWatson(kernel='gaussian', h=i)
    model.fit(Xtest,ytest)

    y_pred3 = model.predict(Xtest)
    
    plt.plot(zz,y_pred3,label='Kernel regression height = %1.4f' % (i))

plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1,label='data')
plt.xlabel("z")
plt.ylabel("$\mu$")    
plt.legend(loc='lower right', fontsize='x-small')
plt.xlim(0,2)
plt.ylim(35,50)


#%%

d = np.arange(2, 21)
training_err = np.zeros(d.shape)
crossval_err = np.zeros(d.shape)

x, x_cv, ydy, ydy_cv = train_test_split(X, np.array([y,dmu]).T, test_size=0.3, random_state=42)
y,dy=ydy.T
y_cv,dy_cv=ydy_cv.T



fig = plt.figure(figsize=(12, 10))
for i in range(len(d)):
    model = PolynomialRegression(i)
    model.fit(x,y,dy)
    
    training_err[i] = np.sqrt(np.sum((y - model.predict(x)) ** 2)/ len(y))
    crossval_err[i] = np.sqrt(np.sum((y_cv - model.predict(x_cv)) ** 2)/ len(y_cv))


plt.plot(d,training_err, label='training set')
plt.plot(d,crossval_err,label='Validation set')
plt.xlabel('degree')
plt.ylabel('Error')
plt.legend(loc=0)

#%%

plt.figure()
classifier = PolynomialRegression(4)

classifier.fit(z_sample[:,np.newaxis], mu_sample,dmu) # in this case dy is irrelevant because errors are homoscedastic
mu_fit = classifier.predict(zz[:,np.newaxis])

mu_fit = classifier.predict(zz[:,np.newaxis])

plt.plot(zz, mu_fit, label='Fit using all data')


classifier.fit(x, y, dy) # in this case dy is irrelevant because errors are homoscedastic
mu_fit = classifier.predict(zz[:,np.newaxis])

plt.plot(zz, mu_fit, label='Fit using only traning data')

#plt.errorbar(z_sample, mu_sample, dmu, fmt='.k', ecolor='gray', lw=1)

plt.errorbar(x, y, dy, fmt='.k', ecolor='gray', lw=1,label='Training data')
plt.errorbar(x_cv, y_cv, dy_cv, fmt='.r', ecolor='red', lw=1, label='Validation data')

#plt.plot(zz, mu_true, '--', c='gray',label='true')

plt.xlabel("z")
plt.ylabel("$\mu$")
plt.xlim(0,2)
plt.ylim(35,50)
plt.legend(loc='lower right')






