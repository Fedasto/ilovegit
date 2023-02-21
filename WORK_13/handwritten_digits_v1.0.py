#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:16:20 2023

@author: federicoastori
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.manifold import Isomap
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#%%

# function to plot the results

def plot_embedding(X, title):
    _, ax = plt.subplots()
    X = MinMaxScaler().fit_transform(X)

    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")
    
    
# import data

digits = datasets.load_digits(n_class=10) 
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n_neighbors = 30

# plot the first 100 digits of that dataset

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
_ = fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)
    
  
isomap_embedding = Isomap(n_neighbors=n_neighbors, n_components=2)
y_pred = isomap_embedding.fit_transform(X, y)


plot_embedding(y_pred, 'Isomap embedding')

plt.show()



#%%

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = LogisticRegression(random_state=0, solver='sag')

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
ypred = clf.predict(X_train)

test_accuracy = metrics.accuracy_score(y_test, predicted)
train_accuracy = metrics.accuracy_score(y_train, ypred)

print('Accuracy on test subsample = %1.3f\n Accuracy on train subsample = %1.3f\n' % (test_accuracy,train_accuracy) )
print('Confusion matrix on train: \n',confusion_matrix(y_train, ypred))
print('confusion matrix on test: \n',confusion_matrix(y_test, predicted))

# plot the confusion matrix

confusion_matrix_train = confusion_matrix(y_train, ypred)
confusion_matrix_test = confusion_matrix(y_test, predicted)

  
cm_display_train = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_train)#, display_labels = [False, True])
cm_display_test = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_test)#, display_labels = [False, True])

cm_display_test.plot()
cm_display_train.plot() 

plt.show()

# The rows represent the actual classes the outcomes should have been. 
# While the columns represent the predictions we have made. Using this table 
# it is easy to see which predictions are wrong.

# e.g. nel train riconosce meglio l'1 rispetto al 8, and confuse 3 with 8 

# Su https://www.w3schools.com/python/python_ml_confusion_matrix.asp ci sono un sacco di metodi aggiuntivi per comprendere meglio la classificazione

#%%

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(np.array(X_test[i].reshape(8, 8)), cmap='binary')
    ax.text(0.05, 0.05, str(predicted[i]), transform=ax.transAxes, 
            color='green' if (y_test[i] == predicted[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])

