#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:43:58 2023

@author: federicoastori
"""

import numpy as np 
import matplotlib.pyplot as plt
import h5py
from astroML.utils import split_samples

from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from astroML.classification import GMMBayes
if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=False)

f = h5py.File('sample_2e7_design_precessing_higherordermodes_3detectors.h5', 'r')
print(list(f.keys()))

X = f['snr']
y = f['det']

#%%

# Let me choose the best classifier for this dataset

# split into training and test sets
(X_train, X_test), (y_train, y_test) = split_samples(X[0:1000], y[0:1000], [0.7, 0.3],random_state=42)
'''
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)

def compute_classification(classifier,args):
    
    for model in classifier:
        
        clf=model
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)
          
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1], pos_label=1)
        
        
        plt.plot(fpr,tpr,label='classifier')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.title('ROC curve')
        plt.xlim(0,0.1)
        plt.ylim(0.9,1)
        plt.legend(loc=0, fontsize='x-small')

models = [GaussianNB(), GMMBayes, QuadraticDiscriminantAnalysis(), DecisionTreeClassifier, LogisticRegression, KNeighborsClassifier]
args = [None, 1, None, 1, 'balanced', 1] 
        
compute_classification(models, args)
'''




    
        
        
        
        
        
    