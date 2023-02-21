#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:05:35 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import h5py
from astroML.utils import split_samples
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import corner 
from sklearn import metrics
from sklearn.metrics import roc_curve
from astroML.utils import split_samples, completeness_contamination
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
#%%


plt.style.use('ggplot')


f = h5py.File('sample_2e7_design_precessing_higherordermodes_3detectors.h5', 'r')
print(list(f.keys()))

snr = f['snr']
det = f['det']
z = f['z']
mtot = f['mtot']
iota = f['iota']
psi = f['psi']
q = f['q']



#%%
lim=10000
X=np.array([snr[:lim],z[:lim],mtot[:lim], iota[:lim], psi[:lim], q[:lim]]).T
index = np.isfinite(X)
X[index==False]=np.mean([X[(index==False)+1],X[(index==False)-1]])
y=det[:lim]

corner.corner(X,labels=['snr','z','mtot', 'iota', 'psi', 'q'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.67)

kfold = model_selection.KFold(n_splits=10, random_state=None)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
scoring = 'accuracy'
for name, model in tqdm(models):
     kfold = model_selection.KFold(n_splits=10, random_state=None)
     cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
     results.append(cv_results)
     names.append(name)
     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
     print(msg)
     
fig = plt.figure(figsize=(10,10))
fig.suptitle('Comparison of sklearn classification algorithms')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.ylim(0.81,1.05)
plt.show()

#%%

# Let me use DecisionTreeClassifier and RandomForset

Ncomponents = np.arange(1,20,2)
classifiers=[]
predictions=[]

plt.figure(figsize=(20,8))
for nc in Ncomponents:
    clf = RandomForestClassifier(max_depth=nc, random_state=0, criterion='entropy')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test[:,:6])
    
    y_prob = clf.predict_proba(X_test[:,:6])[:,1]
    
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    
    predictions.append(y_pred)
    
    plt.subplot(131)
    plt.plot(fpr,tpr,label=' %1.0f components' %(nc))
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC curve')
    plt.xlim(-0.1,0.8)
    plt.ylim(0.8,1.1)
    plt.legend(loc=0, fontsize='x-small')
    
completeness, contamination = completeness_contamination(predictions, y_test)


plt.subplot(132)
plt.title('RandomForest Classifier')
plt.plot(Ncomponents,completeness,'-r', label='completness test set')
plt.plot(Ncomponents,contamination,'-b',label='contamination test set')

plt.xlabel('Ncomponents')
plt.legend(loc=0, fontsize='xx-small')

#%%

C = confusion_matrix(y_test, y_pred)
cm_display_train = metrics.ConfusionMatrixDisplay(confusion_matrix = C)
cm_display_train.plot() 










