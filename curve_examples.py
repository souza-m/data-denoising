# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:13:46 2023

@author: souza-m
"""

import numpy as np
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import pickle
from cycler import cycler

import wkm


# --- data set - step graph ---

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


# -- problem --

# 1. bounded length

m = n
x_list = []

_compute = False
_dump = False
_load = True
_path = './model_dump/bounded_length_step_x_list.pickle'

if _compute:
    parameter = [.1, .125, .15, .175, .2]
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


# 2. bounded curvature

m = n
x_list = [x_pca]

_compute = False
_dump = False
_load = True
_path = './model_dump/bounded_curvature_step_x_list.pickle'

if _compute:
    parameter = [.0005, .0002, .0001, .00004]
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


# -- plots -- 

color_cycle = ['#E7D046', '#1965B0', '#DC050C', '#F1932D', '#4EB265', '#F6C141'] # https://personal.sron.nl/~pault/#fig:scheme_rainbow_discrete -- numbers 10, 15, 18, 20, 26
pl.rcParams['text.usetex'] = True
cc = cycler('color', color_cycle)



# select a few curves to be plotted
select_parameter = [parameter[i] for i in [0, 2, 4]]
select_x_list = [x_list[i] for i in [0, 2, 4]]

# show
fig, ax = pl.subplots(figsize=[7, 5])
ax.axis('equal')
ax.set_prop_cycle(cc)
ax.scatter(x=y[:,0], y=y[:,1], s=13, marker='s', color='black', alpha=.25)
for x in select_x_list:
    ax.plot(x[:,0], x[:,1], linewidth=2.5, alpha=.85)
ax.legend(['Data points'] + [f'R = {p:0.2f}' for p in select_parameter])   # length
ax.legend(['Data points'] + [f'R = {p:0.2f}' for p in select_parameter])
# ax.legend(['Data points', 'PCA'] + [f'B = {p:0.0E}' for p in select_parameter[1:]])   # curvature
# for x in _x_list:
#     ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.75)   # optional


# subplots with both examples side by side

# first run ex. 1 and this
x_list0 = x_list.copy()
x_list0 = [x_list0[i] for i in [0, 2, 4]]
parameter0 = parameter.copy()
parameter0 = [parameter0[i] for i in [0, 2, 4]]

# then run ex. 2 and this
x_list1 = x_list.copy()
x_list1 = [x_list1[i] for i in [0, 2, 4]]
parameter1 = parameter.copy()
parameter1 = [parameter1[i] for i in [0, 2, 4]]

# now plot both copies
fig, ax = pl.subplots(1, 2, sharey=False, figsize=[10, 4])
ax[0].axis('equal')
ax[0].set_prop_cycle(cc)
ax[0].scatter(x=y[:,0], y=y[:,1], s=13, marker='s', color='black', alpha=.25)
for x in x_list0:
    ax[0].plot(x[:,0], x[:,1], linewidth=2.5, alpha=.85)
ax[0].legend(['Data points'] + [f'B = {p:0.2f}' for p in parameter0])   # length

ax[1].axis('equal')
ax[1].set_prop_cycle(cc)
ax[1].scatter(x=y[:,0], y=y[:,1], s=13, marker='s', color='black', alpha=.25)
for x in x_list1:
    ax[1].plot(x[:,0], x[:,1], linewidth=2.5, alpha=.85)
ax[1].legend(['Data points', 'PCA'] + [f'$\lambda$ = {p:0.0E}' for p in parameter1[1:]])   # curvature
# fig.tight_layout()