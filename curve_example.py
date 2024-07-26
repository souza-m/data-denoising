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

# reorder x and pi
order = np.argsort(f[:,0])
x_pca = x_pca[order]
y = y[order]

x_sample = [x_pca]

# plot pca
m = n
x = x_pca.copy()
pi = np.eye(n) / n
plot_transport = True
fig, ax = pl.subplots(figsize=[10, 6])
ax.set_title('PCA')
ax.axis('equal')
ax.scatter(x=x[:,0], y=x[:,1], color='red', s=12, alpha=.5)
ax.plot(x[:,0], x[:,1], color='red', alpha=.25)
ax.scatter(x=y[:,0], y=y[:,1], color='black', s=12, marker='s', alpha=.25)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())


fig, ax = pl.subplots(figsize=[10, 6])
ax.set_title('PCA')
ax.axis('equal')
ax.scatter(x=x[:,0], y=x[:,1], color='#348ABD', s=12, alpha=.5)
ax.plot(x[:,0], x[:,1], color='#348ABD', alpha=.25)
ax.scatter(x=y[:,0], y=y[:,1], color='black', s=12, marker='s', alpha=.25)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())


m = n                   # example 1: m = n

x = x_pca.copy()
pi = np.eye(m) / m
penalty = .05
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x, pi0=pi, epochs=5000, verbose=True, curve_penalty=penalty)
x_sample.append(x)

x = x_pca.copy()
pi = np.eye(m) / m
penalty = .002
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x, pi0=pi, epochs=5000, verbose=True, curve_penalty=penalty)
x_sample.append(x)

# orthogonality condition test
e = sum(pi[i,j] * np.dot(x[i], (y[j] - x[i])) for i in range(m) for j in range(n))
print(f'E[ X . epsilon ] = {e:0.8f}')

# plot individual curve
fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title(f'Curve, penalty = {penalty}')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=12, color='red', alpha=.5)
ax.plot(x[:,0], x[:,1], color='red', alpha=.5)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())

# plot pca and two curves, no transport lines
fig, ax = pl.subplots(figsize=[8, 4])
ax.set_prop_cycle(cc)
ax.set_title('Curves and PCA')
ax.axis('equal')
for x in x_sample:
    # ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
    ax.plot(x[:,0], x[:,1], alpha=.9)
ax.legend(['PCA', 'penalty = 0.05', 'penalty = 0.002'])
for x in x_sample:
    ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
ax.scatter(x=y[:,0], y=y[:,1], s=6, marker='s', color='black', alpha=.32)


# example 2 - m < n
m = 20

# initial position derived from the pca projected points, but reduced
# start with ordered x_pca and join centroids to reduce number of x-points from n to m
pi = np.zeros([m, n])
max_row = 1/m
max_col = 1/n
i = 0
j = 0
total_row = 0.
total_col = 0.
total = 0.
while total < 1 - 1e-6:
    new_w = min(max_row - total_row, max_col - total_col)
    pi[i,j] = new_w
    total_row += new_w
    total_col += new_w
    total += new_w
    if np.isclose(total_row, max_row):
        # step down
        i += 1
        total_row = 0.
    else:
        # step right
        j += 1
        total_col = 0.
bary = x_pca.mean(axis=0)
x = bary + np.dot(pi, x_pca - bary) * n / m

x_sample_low_m = [x]

# first initialize x and pi above
penalty = .002
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x_sample_low_m[0], pi0=pi, epochs=5000, verbose=True, curve_penalty=penalty)
x_sample_low_m.append(x)


# first initialize x and pi above
penalty = .05
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x_sample_low_m[0], pi0=pi, epochs=5000, verbose=True, curve_penalty=penalty)
x_sample_low_m.append(x)






# plot pca and two curves, no transport lines
fig, ax = pl.subplots(figsize=[10, 6])
ax.set_title('Curves and PCA')
ax.axis('equal')
for x in x_sample_low_m:
    # ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
    ax.plot(x[:,0], x[:,1], alpha=.5)
ax.legend(['PCA', 'penalty = 0.002', 'penalty = 0.05'])
for x in x_sample_low_m:
    ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)

# red pattern (presentation consistency)
fig, ax = pl.subplots(figsize=[7, 4])
ax.set_title('Curves and PCA')
ax.axis('equal')
ax.plot(x_sample[0][:,0], x_sample[0][:,1], color='red', alpha=.5)
for x in x_sample[1:][::-1]:
    # ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
    ax.plot(x[:,0], x[:,1], alpha=.5)
ax.legend(['PCA', 'penalty = 0.05', 'penalty = 0.002'])
ax.scatter(x=x_sample[0][:,0], y=x_sample[0][:,1], color='red', s=12, alpha=.5)
for x in x_sample[1:][::-1]:
    ax.scatter(x=x[:,0], y=x[:,1], s=12, alpha=.5)
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)





# plot initial position
# looks shifted left...
fig, ax = pl.subplots(figsize=[10, 6])
ax.set_title('PCA')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=22, color='red', alpha=.99)
ax.plot(x[:,0], x[:,1], color='red', alpha=.25)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())

# fit
penalty = .02
x, pi, exy_series = wkm.fit(y, m, 'curve', x0=x, pi0=pi, epochs=500, verbose=True, curve_penalty=penalty)

# plot
fig, ax = pl.subplots(figsize=[8, 6])
ax.set_title(f'Curve, penalty = {penalty}')
ax.axis('equal')
ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
ax.scatter(x=x[:,0], y=x[:,1], s=12, color='red', alpha=.5)
ax.plot(x[:,0], x[:,1], color='red', alpha=.5)
for i in range(m):
  for j in range(n):
      ax.plot((x[i,0], y[j,0]), (x[i,1], y[j,1]), color='green', alpha=.5, linewidth=20*pi[i,j]*0.1/pi.max())





# # self-consistent PCA

# m = n
# x = x_pca
# a = pca.components_[0]
# A = np.dot(a.reshape([len(a), 1]), a.reshape([1, len(a)])) / (np.linalg.norm(a) ** 2)   # projection matrix

# # # centralize y
# # barycenter = y.mean(axis=0)
# # yc = y - barycenter

# # # we are interested in the projection of y to the principal axis
# # yp = np.vstack([np.dot(A, yc[j,:]) for j in range(n)])
# # yp = yp + barycenter

# x, pi, exy_series = wkm.fit(y, m, 'self_consistent_pca', epochs=20, verbose=True, principal_direction=a)
# fig, ax = pl.subplots(figsize=[8, 6])
# ax.set_title('Self-consistent PCA')
# ax.axis('equal')
# ax.scatter(x=y[:,0], y=y[:,1], s=12, marker='s', color='black', alpha=.25)
# ax.scatter(x=x[:,0], y=x[:,1], s=22, color='red', alpha=.99)
# ax.plot(x[:,0], x[:,1], color='red', alpha=.25)
