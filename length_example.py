# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:46 2023

@author: souza-m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import wkm

from cycler import cycler
cc = cycler('color', ['#A60628', '#348ABD', '#D55E00', '#467821', '#7A68A6', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2'])

import pickle

# --- data set -- choose synthetic (step graph) or real data (air quality data) ---

# step graph

def rho_step_2d(n):
    third = int(n/3)
    t = np.concatenate([np.linspace(-1, 0, third), np.zeros(n - 2 * third), np.linspace(0, 1, third)])
    s = np.concatenate([np.zeros(third), np.linspace(0, 1, n - 2 * third), np.ones(third)])
    p = np.vstack([t, s]).T
    return p

d = 2
n = 500
m = 400

np.random.seed(1)
noise_var = .01
p = rho_step_2d(n)
mean = [0 for i in range(d)]
cov = noise_var * np.eye(d)
y = np.vstack([p[i,:] + np.random.multivariate_normal(mean, np.random.random(d) * cov, size=1) for i in range(n)])
    
pca = PCA(n_components=1)
f = pca.fit_transform(y)
x_pca = pca.inverse_transform(f)   # pca.mean_ + f * pca.components_
order = np.argsort(f[:,0])
x_pca = x_pca[order]

length_parameter = [.1, .125, .15, .175, .2]
curvature_parameter = [.0005, .0002, .0001, .00004]


# air quality

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
n = len(y)
m = n

length_parameter = [.1, .25, .4]
curvature_parameter = [.05, .002, .001]


# -- problem -- choose bounded curvature or bounded length --

# bounded curvature
m = n
parameter = curvature_parameter
x_list = [x_pca]
for penalty in parameter:
    # x, pi, exy_series = wkm.fit(y, m, 'curvature', x0=x0, pi0=pi, epochs=500, verbose=True, curvature_penalty=penalty)
    x, pi, exy_series = wkm.fit(y, m, 'curvature', x0=x_pca, pi0 = np.eye(m) / m, epochs=5000, verbose=False, curvature_penalty=penalty)
    x_list.append(x)

# bounded length

m = n
parameter = length_parameter
x_list = []
for B in parameter:
    x, pi, exy_series = wkm.fit(y, m, 'length', x0=x_pca, pi0 = np.eye(m) / m, epochs=5, verbose=False, length=B)
    x_list.append(x)

old_x_list = x_list.copy()
# -- plot -- 

_parameter = parameter
_x_list = x_list
# _x_list = x_sample
# _parameter = [parameter[0], parameter[2], parameter[4]]
# _x_list = [x_list[0], x_list[2], x_list[4]]
fig, ax = pl.subplots(figsize=[8, 5])
ax.axis('equal')
ax.set_prop_cycle(cc)
for x in _x_list:
    ax.plot(x[:,0], x[:,1], linewidth=2.5, alpha=.85)
# ax.legend(['PCA'] + [f'{p:0.1f}' for p in _parameter])
ax.legend([f'{p:0.2f}' for p in _parameter])
# for x in _x_list:
#     ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.75)
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)

x_list[0][:5]
x_list[1][:5]
x_list[2][:5]
# k-means w/ fixed weights
# m = 40   # air
m = 100   # step
x, pi, exy_series = wkm.fit(y, m, 'kmeans_fixed', epochs=5, verbose=True)
order = np.argsort(x.sum(axis=1))    # manual ordering
x = x[order]
x_list = [x]
u = np.ones(m) / m   # weights

# reguar k-means (free weights) -- Lloyd's
kmeans = KMeans(n_clusters=m, random_state=0).fit(y)
kc = kmeans.predict(y)
uk = 1. * np.array([(kc == i).sum() for i in set(kc)])
uk /= uk.sum()   # weights
x_km = kmeans.cluster_centers_

# -- plot -- 
# _parameter = parameter
_x_list = x_list
fig, ax = pl.subplots(figsize=[8, 5])
ax.axis('equal')
ax.set_prop_cycle(cc)
for x in _x_list:
    ax.scatter(x=x[:,0], y=x[:,1], s= 700 * u, alpha=.75)
    ax.scatter(x=x_km[:,0], y=x_km[:,1], s= 1700 * uk, alpha=.75)
ax.legend(['fixed weights', 'free weights'])
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=p[:,0], y=p[:,1], s=4, marker='s', color='yellow')
ax.set_prop_cycle(cc)
for x in _x_list:
    ax.scatter(x=x[:,0], y=x[:,1], s= 700 * u, alpha=.5)
    ax.scatter(x=x_km[:,0], y=x_km[:,1], s= 1700 * uk, alpha=.5)


# -- save --

DF = {'x_list': x_list, 'parameter': parameter}
_path = './model_dump/bounded_length_step_x_list.pickle'
# with open(_path, 'wb') as file:
#     pickle.dump(DF, file)
# print('x_list and parameter saved to ' + _path)

with open(_path, 'rb') as file:
    DF = pickle.load(file)
