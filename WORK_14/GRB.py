#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:50:18 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from astropy.stats import freedman_bin_width
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%

# Download file
r = requests.get('https://user-web.icecube.wisc.edu/~grbweb_public/Summary_table.txt')
with open("Summary_table.txt", 'wb') as f:
    f.write(r.content)

# Read content
data = np.loadtxt("Summary_table.txt", dtype='str',unpack='True')

# Read headers
with open("Summary_table.txt",'r') as f:
    names= np.array([n.strip().replace(" ","_") for n in f.readlines()[1].replace("#","").replace("\n","").lstrip().split('    ') if n.strip()!=''])

# T90=np.array(data[6],dtype=float)
# print(T90)

T0 = np.array(data[3],dtype=float)
t_trigger = np.array(data[4],dtype=float)
ra = np.array(data[5],dtype=float)
decl=ra = np.array(data[6],dtype=float)
pos_error =np.array(data[7],dtype=float)
#T90 = np.array(data[8],dtype=float)
T90_error = np.array(data[9],dtype=float)
T90_start = np.array(data[10],dtype=float)
fluence = np.array(data[11],dtype=float)
fluence_error = np.array(data[12],dtype=float)
#redshift = np.array(data[13],dtype=float)
T100 = np.array(data[14],dtype=float)

# data_set = [T0, t_trigger, ra, decl, pos_error, T90_error, T90_start, fluence, fluence_error,T100]
data_set = [T0, t_trigger, ra, decl, T90_start, fluence, T100]


#%%

# 1. Presence of Clusters
# 1.1 KDE cross validation

T90=np.array(data[6],dtype=float)
x=T90

def kde_sklearn(data, bandwidth = 1.0):
    kde_skl = KernelDensity(bandwidth = bandwidth)
    kde_skl.fit(data[:, np.newaxis])
    log_pdf = kde_skl.score_samples(xgrid[:, np.newaxis]) # sklearn returns log(density)
    return np.exp(log_pdf)

bwrange = np.linspace(0.01,1.0, 30) # Test 30 bandwidths from 0.1 to 1.0
# print(bwrange)
K = 5 # Do 5-fold cross validation
grid = GridSearchCV(KernelDensity(), {'bandwidth': bwrange}, cv= K) # Try each bandwidth with K-folds
grid.fit(x[:, None]) #Fit the histogram data that we started the lecture with.
h_opt = grid.best_params_['bandwidth']
print('Optimal bandwidth: ',h_opt)


width, bins = freedman_bin_width(T90, return_bins=True)
plt.hist(T90,bins=bins,density=True, label='T90')
xgrid =np.linspace(np.min(bins),np.max(bins),100)
pdf = kde_sklearn(x,bandwidth=h_opt)
plt.plot(xgrid,pdf,c='red', label='PDF', linewidth=0.5)
plt.title('KDE classification')
plt.legend(loc=0)

#%%

# 1.2 Mean Shift clustering



pca = PCA(n_components=2) # 2 components
pca.fit(data_set) # Do the fitting

X_reduced = pca.transform(data_set)

plt.figure(figsize=(8,8))
plt.scatter(X_reduced[:,0], X_reduced[:,1], marker="o", color='C0', alpha=0.01, edgecolors='None')
plt.xlabel('Eigenvalue 1')
plt.ylabel('Eigenvalue 2')


scaler = preprocessing.StandardScaler()
bandwidth = 1.0
#bandwidth = estimate_bandwidth(X_reduced) # this takes a long time...beware
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
ms.fit(scaler.fit_transform(X_reduced))

labels_unique = np.unique(ms.labels_)
n_clusters = len(labels_unique[labels_unique >= 0])
print(labels_unique)
print(bandwidth)
print("number of estimated clusters :", n_clusters)

# Make some plots
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

# Compute a 2D histogram  of the input
H, xedges, yedges = np.histogram2d(X_reduced[:,0], X_reduced[:,1], 50)

# plot density
plt.imshow(H.T, origin='lower', interpolation='nearest', aspect='auto',
          extent=[xedges[0], xedges[-1],
                  yedges[0], yedges[-1]],
          cmap='Blues')

# plot cluster centers
cluster_centers = scaler.inverse_transform(ms.cluster_centers_)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           s=40, c='w', edgecolors='k')

# plot cluster boundaries
x_centers = 0.5 * (xedges[1:] + xedges[:-1])
y_centers = 0.5 * (yedges[1:] + yedges[:-1])

Xgrid = np.meshgrid(x_centers, y_centers)
Xgrid = np.array(Xgrid).reshape((2, 50 * 50)).T

H = ms.predict(scaler.transform(Xgrid)).reshape((50, 50))

for i in range(n_clusters):
    Hcp = H.copy()
    flag = (Hcp == i)
    Hcp[flag] = 1
    Hcp[~flag] = 0

    plt.contour(x_centers, y_centers, Hcp, [-0.5, 0.5],
               linewidths=1, colors='k')
 
    H = ms.predict(scaler.transform(Xgrid)).reshape((50, 50))
    
ax.set_xlim(xedges[0], xedges[-1])
ax.set_ylim(yedges[0], yedges[-1])

ax.set_xlabel('Eigenvalue 1')
ax.set_ylabel('Eigenvalue 2')

plt.show()

# identified at least 2 subsamples working on the bandwidth

#%%

# 1.3 K-means

X=data_set

clf = KMeans(n_clusters=2) #Try 2 clusters to start with
clf.fit(X)
centers = clf.cluster_centers_ #location of the clusters
labels = clf.predict(X) #labels for each of the points

# Compute the KMeans clustering
n_clusters = 3
scaler = preprocessing.StandardScaler()
clf = KMeans(n_clusters)
clf.fit(scaler.fit_transform(X_reduced))

# Make some plots
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()

# Compute a 2D histogram  of the input
H, xedges, yedges = np.histogram2d(X_reduced[:,0], X_reduced[:,1], 50)

# plot density
ax.imshow(H.T, origin='lower', interpolation='nearest', aspect='auto',
          extent=[xedges[0], xedges[-1],
                  yedges[0], yedges[-1]],
          cmap='Blues')

# plot cluster centers
cluster_centers = scaler.inverse_transform(clf.cluster_centers_)
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
           s=40, c='w', edgecolors='k')

# plot cluster boundaries
x_centers = 0.5 * (xedges[1:] + xedges[:-1])
y_centers = 0.5 * (yedges[1:] + yedges[:-1])

Xgrid = np.meshgrid(x_centers, y_centers)
Xgrid = np.array(Xgrid).reshape((2, 50 * 50)).T

H = clf.predict(scaler.transform(Xgrid)).reshape((50, 50))

for i in range(n_clusters):
    Hcp = H.copy()
    flag = (Hcp == i)
    Hcp[flag] = 1
    Hcp[~flag] = 0

    ax.contour(x_centers, y_centers, Hcp, [-0.5, 0.5],
               linewidths=1, colors='k')

    H = clf.predict(scaler.transform(Xgrid)).reshape((50, 50))
    
ax.set_xlim(xedges[0], xedges[-1])
ax.set_ylim(yedges[0], yedges[-1])

ax.set_xlabel('Eigenvalue 1')
ax.set_ylabel('Eigenvalue 2')

plt.show()

# if n_cluster is ==3 I find the same result of before, whle with ==2 I obtain a treshold value around 2 in egenvalue 2.

#%%

# 1.4 Gaussian mixture




