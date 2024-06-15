# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:46 2023

@author: souza-m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import wkm


# example 1 - principal curve with bounded curvature, airquality data
air = pd.read_csv('data/airquality.csv')
air = air[['Ozone', 'Temp']].dropna()
y = air.values
pca = PCA(n_components=1)
f = pca.fit_transform(y)
x = pca.inverse_transform(f)   # pca.mean_ + f * pca.components_
n = m = len(y)

# reorder x and pi
order = np.argsort(f[:,0])
x = x[order]
y = y[order]
pi = np.eye(m) / m

# plot pca
plot_transport = False
fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title('PCA')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=12, color='red', alpha=.5)
ax.plot(x[:,0], x[:,1], color='red', alpha=.25)
if plot_transport:
    for i in range(m):
      for j in range(n):
          ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())

# denoise
theta = 4.
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x, pi0=pi, epochs=200, verbose=True, theta=theta)

# plot curve
fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title('Curve, theta = {theta}')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=12, color='red', alpha=.5)
ax.plot(x[:,0], x[:,1], color='red', alpha=.25)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())