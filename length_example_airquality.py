# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:26:20 2024

@author: marce
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import wkm

from cycler import cycler
cc = cycler('color', ['#A60628', '#348ABD', '#D55E00', '#467821', '#7A68A6', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])


# principal curve examples - bounded curvature


# air quality data
air = pd.read_csv('data/airquality.csv')
air = air[['Ozone', 'Temp']].dropna()
y = air.values
pca = PCA(n_components=1)
f = pca.fit_transform(y)
x_pca = pca.inverse_transform(f)   # pca.mean_ + f * pca.components_
n = len(y)

# reorder x and pi in the order of the principal axis
order = np.argsort(f[:,0])
x_pca = x_pca[order]
y = y[order]

# build a set of solutions with increasing curvature
# start with pca
x_sample = [x_pca]

# plot pca
m = n
x = x_pca.copy()
pi = np.eye(n) / n
plot_transport = True




# bounded length
m = n
B=.4

fig, ax = pl.subplots(figsize=[8, 6])
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
for B in [.2, .3, .4, .5]:
    x0 = x_pca.copy()
    pi0 = np.eye(m) / m
    x, pi, exy_series = wkm.fit(y, m, 'bounded_length', x0=x0, pi0=pi0, epochs=5, verbose=False, length=B)
    
    # plot
    ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
    ax.plot(x[:,0], x[:,1], alpha=.5)
# for i in range(m):
#   for j in range(n):
#       ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())