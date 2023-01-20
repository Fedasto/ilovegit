#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:41:34 2023

@author: federicoastori
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

#%%

M = 1000
x = stats.uniform(0,1).rvs(M) # M random draws between 0 and 1
y = stats.uniform(0,1).rvs(M) # M random draws between 0 and 1
z = stats.uniform(0,1).rvs(M) # M random draws between 0 and 1

r3 = x**2+y**2+z**2 # equation for radius of cirle in x,y,z

# set the canvas

fig = plt.figure()
ax= fig.add_subplot(projection='3d')

ax.scatter(x,y,z, marker='.', color='blue', s=3)

ax.scatter(x[r3 < 1], y[r3 < 1], z[r3 < 1], marker='x', color='red', s=3)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# volume_cube= d^N; volume_sphere=4/3*pi*d^N

V_Est = 4/3*np.pi*np.sum(r3<1)/M
err_V_Est = np.abs(((4/3*np.pi)-V_Est) / 4/3*np.pi)
print('the estimate is %1.3f \n the # of draws is %1.0f \n the absolute error is %1.3f \n' % (V_Est, M, err_V_Est))

#%%
M=1000
def ND_sphere(N_dim):
    par=np.zeros([M,N_dim])
    r=np.zeros(M)
    for i in range(N_dim):
        for j in range(M):
            par[j][i]=stats.uniform(0,1).rvs(1)
    for t in range(M):
        for k in range(N_dim):
            r[t]=np.sum(par[t][k]**2)
    
    return ((4/3*np.pi*np.sum(r<1)/M), np.abs(((4/3*np.pi)-(4/3*np.pi*np.sum(r<1)/M)) / 4/3*np.pi))

estimate, error = ND_sphere(3)
print('estimate=%1.3f' % (estimate))
print('error=%1.3f' % (error))

#%%

dim = np.linspace(4,10,3) # numero di dimensioni non M!!!

xx=[]
er=[]
for k in tqdm(dim):
    v,err = ND_sphere(int(k))
    xx.append(v)
    er.append(err)
    
plt.figure()    
plt.bar(dim,xx, alpha=0.5)
plt.errorbar(dim,xx,er, capsize=2 )
plt.xlabel('# dimensions')
plt.ylabel('pdf')

    
    
    


