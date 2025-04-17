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
import pickle
from cycler import cycler

import wkm


# --- data set -- choose synthetic (step graph) or real data (air quality data) ---

# 1. step graph

def rho_step_2d(n):
    third = int(n/3)
    t = np.concatenate([np.linspace(-1, 0, third), np.zeros(n - 2 * third), np.linspace(0, 1, third)])
    s = np.concatenate([np.zeros(third), np.linspace(0, 1, n - 2 * third), np.ones(third)])
    p = np.vstack([t, s]).T
    return p

d = 2
n = 300
# m = 500

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


# 2. air quality (not used)

# air = pd.read_csv('data/airquality.csv')
# air = air[['Ozone', 'Temp']].dropna()
# y = air.values
# pca = PCA(n_components=1)
# f = pca.fit_transform(y)
# x_pca = pca.inverse_transform(f)   # pca.mean_ + f * pca.components_
# n = len(y)

# # reorder x and pi in the order of the principal axis
# order = np.argsort(f[:,0])
# x_pca = x_pca[order]
# y = y[order]
# n = len(y)
# m = n

# length_parameter = [.1, .25, .4]
# curvature_parameter = [.05, .02, .01, .005, .002, .001]
# curvature_parameter = [.001, .0002, .0001, .00002, .00001]


# -- problem -- choose bounded curvature or bounded length --

# 1. bounded curvature
m = n
x_list = [x_pca]

_compute = False
_dump = False
_load = True
_path = './model_dump/bounded_curvature_step_x_list.pickle'

if _compute:
    parameter = curvature_parameter
    for penalty in parameter:
        # x, pi, exy_series = wkm.fit(y, m, 'curvature', x0=x0, pi0=pi, epochs=500, verbose=True, curvature_penalty=penalty)
        x, pi, exy_series = wkm.fit(y, m, 'curvature', x0=x_pca, pi0 = np.eye(m) / m, epochs=50000, verbose=False, curvature_penalty=penalty)
        x_list.append(x)

if _dump:
    DF = {'x_list': x_list, 'parameter': parameter}
    with open(_path, 'wb') as file:
        pickle.dump(DF, file)
    print('x_list and parameter saved to ' + _path)

if _load:
    with open(_path, 'rb') as file:
        DF = pickle.load(file)
    print('x_list and parameter loaded from ' + _path)
    x_list = DF['x_list']
    parameter = DF['parameter']



# 2. bounded length
m = n
x_list = []

_compute = False
_dump = False
_load = True
_path = './model_dump/bounded_length_step_x_list.pickle'

if _compute:
    parameter = length_parameter
    for B in parameter:
        x, pi, exy_series = wkm.fit(y, m, 'length', x0=x_pca, pi0 = np.eye(m) / m, epochs=5, verbose=False, length=B)
        x_list.append(x)

if _dump:
    DF = {'x_list': x_list, 'parameter': parameter}
    with open(_path, 'wb') as file:
        pickle.dump(DF, file)
    print('x_list and parameter saved to ' + _path)

if _load:
    with open(_path, 'rb') as file:
        DF = pickle.load(file)
    print('x_list and parameter loaded from ' + _path)
    x_list = DF['x_list']
    parameter = DF['parameter']



# 3. k-means w/ fixed weights (not used)

# # m = 40   # air
# m = 50   # step
# x, pi, exy_series = wkm.fit(y, m, 'kmeans_fixed', epochs=5, verbose=True)
# order = np.argsort(x.sum(axis=1))    # manual ordering
# x = x[order]
# x_list = [x]
# u = np.ones(m) / m   # weights

# # reguar k-means (free weights) -- Lloyd's
# kmeans = KMeans(n_clusters=m, random_state=0).fit(y)
# kc = kmeans.predict(y)
# uk = 1. * np.array([(kc == i).sum() for i in set(kc)])
# uk /= uk.sum()   # weights
# x_km = kmeans.cluster_centers_


# -- plots -- 

color_cycle = ['#E7D046', '#1965B0', '#DC050C', '#F1932D', '#4EB265', '#F6C141'] # https://personal.sron.nl/~pault/#fig:scheme_rainbow_discrete -- numbers 10, 15, 18, 20, 26

# curves

pl.rcParams['text.usetex'] = True
cc = cycler('color', color_cycle)
# _x_list = x_sample
# _parameter = [parameter[0], parameter[2], parameter[4]]
# _x_list = [x_list[0], x_list[2], x_list[4]]
fig, ax = pl.subplots(figsize=[7, 5])
ax.axis('equal')
ax.set_prop_cycle(cc)
ax.scatter(x=y[:,0], y=y[:,1], s=13, marker='s', color='black', alpha=.25)
for x in _x_list:
    ax.plot(x[:,0], x[:,1], linewidth=2.5, alpha=.85)
# ax.legend(['PCA'] + [f'{p:0.1f}' for p in _parameter])
# ax.legend(['data points'] + [f'R = {p:0.2f}' for p in _parameter])
ax.legend(['data points', 'PCA'] + [f'$\lambda$ = {p:0.0E}' for p in _parameter[1:]])
# for x in _x_list:
#     ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.75)


# scatterplot (not used)

# cc = cycler('color', color_cycle[1:])
# _parameter = parameter
# _x_list = x_list
# fig, ax = pl.subplots(figsize=[7, 5])
# ax.axis('equal')
# ax.set_prop_cycle(cc)
# ax.scatter(x=y[:,0], y=y[:,1], s=13, marker='s', color='black', alpha=.25)
# ax.plot(p[:,0], p[:,1], color=color_cycle[0], linewidth=2.5, alpha=.7)
# for x in _x_list:
#     ax.scatter(x=x[:,0], y=x[:,1], s= 1500 * u, alpha=.75)
#     ax.scatter(x=x_km[:,0], y=x_km[:,1], s= 1500 * uk, alpha=.75)
# ax.legend(['data points', 'originating curve', 'centroids with fixed weights', 'centroids with free weights'])
# ax.set_prop_cycle(cc)
# for x in _x_list:
#     ax.scatter(x=x_km[:,0], y=x_km[:,1], s= 1500 * uk, alpha=.8, color=color_cycle[2])
#     ax.scatter(x=x[:,0], y=x[:,1], s= 1500 * u, alpha=.9, color=color_cycle[1])


