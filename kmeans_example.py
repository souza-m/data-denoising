# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:06:41 2024

@author: souzam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import wkm


# k-means examples


# air quality data
air = pd.read_csv('data/airquality.csv')
air = air[['Ozone', 'Temp']].dropna()
y = air.values
pca = PCA(n_components=1)
f = pca.fit_transform(y)
x_pca = pca.inverse_transform(f)   # pca.mean_ + f * pca.components_

n = len(y)
m = int(np.sqrt(n))   # choose m < n

# example 1 - fixed weights

# fit and plot
x, pi, exy_series = wkm.fit(y, 100, 'fixed_u', epochs=5, verbose=True)

fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title('Fixed-weights K-means')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=50, color='red', alpha=.5)

# example 2 - variable weights (traditional)
# this is just a proof of concept
# Lloyd's algorithm performs better in general in this non-penalized problem
# fit and plot with centroid size proportional to its mass

# fit
x, pi, exy_series = wkm.fit(y, m, 'variable_u', epochs=20, verbose=True)
u = pi.sum(axis=1)
wkc, pi = wkm.wk_classify(y, x, pi = pi)

# Lloyd's
kmeans = KMeans(n_clusters=m, random_state=0).fit(y)
kc = kmeans.predict(y)
uk = 1. * np.array([(kc == i).sum() for i in set(kc)])
uk /= uk.sum()
x_km = kmeans.cluster_centers_

# visual comparison
fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title('Variable-weights K-means')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=60*m*u, color='red', alpha=.75)
ax.scatter(x=x_km[:,0], y=x_km[:,1], s=60*m*uk, color='blue', alpha=.75)

# cluster index comparison
wkm.report_indices(kc, wkc, y, None, title = 'synthetic')
