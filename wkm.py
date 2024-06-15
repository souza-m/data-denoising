# -*- coding: utf-8 -*-
"""
Created on Sep-2023

@author: souza-m
"""

import numpy as np
import sinkhorn
from sklearn import metrics

'''
notation:
    y  (n,d) is the noisy data (input)
    x  (m,d) is the denoised set of points (solution)
    xh (m,d) is the auxiliary variable in the unit ellipse
    v is the y-marginal probability distribution
    u is the x-marginal probability distribution
    pi (m,n) is the transport matrix
'''


# --- main function ---

fit_methods = ['fixed_u', 'variable_u', 'curve']
def fit(y, m, method, x0 = None, pi0 = None, epochs = 1, verbose = False, **kwargs):
    
    # initialize and centralize
    n = len(y)
    v = kwargs.get('v', np.ones(n) / n)
    barycenter = np.dot(v, y)
    y -= barycenter
    if not x0 is None:
        x0 -= barycenter
    x = x0
    pi = pi0
    obj_series = []
    
    # principal curve with bounded curvature
    if method == 'curve':
 
        # only accepting uniform weights for the moment (ignoring parameter u)
        u = np.ones(m) / m
        
        # treat parameters
        alpha = .01
        if x0 is None or pi0 is None:
            raise ValueError("initial position and transport plan must be provided")
        if exx(x0) == 0:
            raise ValueError("initial position must have positive variance")
            
        # initial position and transport plan
        xh = x0 / np.sqrt(exx(x0))
        pi = pi0
        
        # iterate
        for epoch in range(epochs):
            xh = update_xh(pi, y, xh, method, alpha, kwargs)
            pi = update_pi(xh, y, v=v, u=u, old_pi=pi, method='nearest')
            
            # report
            x = xh * exy(xh, y, pi)   # rescale
            EXX = exx(x, pi.sum(axis=1))
            EYY = exx(y)
            EXY = exy(x, y, pi)    
            obj_series.append(EXY)
            if verbose and (epochs <= 30 or epoch+1 <= 5 or (epoch+1)%10 == 0):
                r = EXY / np.sqrt(EXX * EYY)
                R2 = r ** 2
                zero_weights = np.isclose(pi.sum(axis=1), 0, atol=1e-5)
                print(f'{epoch+1:4d}     R2 = {R2:6.2%}   # nonzero weights = {m - zero_weights.sum():3d} / {m}')
    
    else:
        print('--- not implemented ---')
    
    # wrap and return
    if epochs == 0:
        x = xh * exy(xh, y, pi)   # rescale
    if not x is None:
        x += barycenter
    return x, pi, obj_series


# --- optimization functions ---

# update xh
xh_methods = ['fixed_u', 'variable_u', 'curve']
def update_xh(pi, y, xh0, method, alpha, **kwargs):
    m, n = pi.shape
    _, d = y.shape
    u = pi.sum(axis=1)
    
    # find new xh
    yhat = np.dot(pi, y)
    if method in ['fixed_u', 'variable_u'] or (method == 'curve' and not 'curve_mult' in kwargs.keys()):
        _xh = ellipse_max(yhat, u, centralized=True)
    elif method == 'curve':
        curve_mult = kwargs['curve_mult']
        penalty = curve_mult * phi(xh0)
        _xh = ellipse_max(yhat + penalty, u)
    else:
        print('--- error: method not identified')
        return
        
    # check barycenter constraint
    xh_bary = u[:,None] * _xh
    assert np.isclose(xh_bary, np.zeros(d), atol=1e-6).all(), 'barycenter condition violated'
        
    # check second moment constraint
    xh_secmon = sum(u * (_xh ** 2).sum(axis=0))
    assert np.isclose(xh_secmon, 1, atol=1e-6), 'second moment condition violated'
    if xh_secmon < 1 - 1e-6:
        print('--- warning: variance less than 1', xh_secmon)
    
    # partial update and return
    xh = _xh if xh0 is None else alpha  * _xh + (1 - alpha) * xh0
    return xh

# min C.x
#  st u.x  == 0
#     u.x2 <= 1
def ellipse_max(C, u, centered = False):
    m, d = C.shape
    
    # treat zero weights, avoid division by zero
    _u = u.copy()
    for i in range(m):
        if np.isclose(u[i], 0):
            _u[i] = np.inf
    u_divisor = 1 / _u
    
    # refer to appendix for Lagrangian
    kappa = C.sum(axis=0)
    if centered:
        assert np.isclose(kappa, 0).all(), 'kappa not centered'
        s = (C ** 2) * u_divisor[:,None]
    else:
        s = (C ** 2) * u_divisor[:,None] - 2 * C * kappa[None,:] + u[:,None] * (kappa ** 2)[None,:]
    lamb = np.sqrt(s.sum()) / 2
    
    xh = (C * u_divisor[:,None] - kappa[None,:]) / (2 * lamb)
    assert np.isclose((u[:,None] * xh).sum(axis=0), np.zeros(d), atol=1e-6).all(), 'error: xh not centered'
    assert np.isclose((u[:,None] * (xh ** 2)).sum(), 1), 'error: var(xh) != 1'
    return xh

# linear function of xh(t) based on xh(t-1)
# measures curvature
def phi(xh0):
    m, d = xh0.shape
    A = [xh0[i+1,:] - xh0[i,:] for i in range(m-1)]
    anorm = [np.linalg.norm(a) for a in A]
    anorm = [norm if norm > 0 else np.infty for norm in anorm]
    E = [A[i+1] / anorm[i+1] - A[i] / anorm[i] for i in range(m-2)]
    D = [np.zeros([1,d])] + [-e for e in E] + [np.zeros([1,d])]
    cpenalty = np.vstack(D)
    return cpenalty

# update pi
pi_methods = ['lp_free_u', 'sinkhorn_free_u', 'sinkhorn_fixed_u', 'nearest']
def update_pi(xh, y, v, u, method, pi0 = None, alpha = 1):
    m, n = len(xh), len(y)
    d, _d = xh.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    
    if method == 'nearest':
        C = distance_matrix(xh, y)
        _pi = nearest_pi(C, pi0)
    elif method == 'sinkhorn_fixed_u':
        C = crossprod_matrix(xh, y)
        # _pi = sinkhorn.sinkhorn_pi(C, v, u)
        print('--- not implemented ---')
        return
    elif method == 'sinkhorn_free_u':
        # C = crossprod_matrix(xh, y)
        # xh_sq = np.sum(xh*xh, axis=1)
        print('--- not implemented ---')
        return
    elif method == 'lp_free_u':
        # C = crossprod_matrix(xh, y)
        # xh_sq = np.sum(xh*xh, axis=1)
        # _pi = lp_pi(C, v, xh, xh_sq, floor = 0, ceil = 1)
        print('--- not implemented ---')
        return
    else:
        print('methods: sinkhorn_fixed, sinkhorn_free, lp_free')
        return
    
    # clean, update and return
    _pi = np.maximum(_pi, 0.)
    _pi = _pi * np.repeat(v / _pi.sum(axis=0), m).reshape([n, m]).T
    pi = _pi if alpha == 1 or pi0 is None else alpha  * _pi + (1. - alpha) * pi0
    return pi

# sigma is the direct assignment
# conversion only makes sense for 1-1 transport matrices
def pi_to_sigma(pi):
    m = pi.shape[0]
    ans = []
    for i in range(m):
        ans.append(pi[i,:].argmax())
    return ans
def sigma_to_pi(sigma):
    m = len(sigma)
    pi = np.zeros([m, m])
    for i, j in enumerate(sigma):
        pi[i, j] = 1/m
    return pi

# assume that sigma0 is close to the solution (otherwise the problem is most likely np-hard)
def best_assignment(C, sigma):
    # NOTE: still not exact, needs a loop until no more swaps
    if len(sigma) == 1:
        return sigma
    m = len(sigma)
    # swap first assignment?
    for i in range(1, m):
        if C[0, sigma[i]] + C[i, sigma[0]] < C[0, sigma[0]] + C[i, sigma[i]]:
            # print('--- swap', sigma[0], sigma[i])
            sigma[0], sigma[i] = sigma[i], sigma[0]
            return best_assignment(C, sigma)
    # _C = np.delete(C[1:,:], sigma[0], 1)
    return [sigma[0]] + best_assignment(C[1:,:], sigma[1:])

# find exact optimal pi
# pi0 must be composed of 0's and (1/n)'s
def nearest_pi(C, pi0):
    # return pi0
    return sigma_to_pi(best_assignment(C, pi_to_sigma(pi0)))


# --- external utility functions ---

# classification by nearest-centroid criterion
def wk_classify(y, xh, pi = None):
    n, d = y.shape
    m, _d = xh.shape
    assert d == _d, 'dimension mismatch'
    
    if pi is None:
        pi = update_pi(xh, y)
    classes = np.array(range(m))
    wk = n * pi.T.dot(classes)
    wk = np.array([int(c) for c in np.round(wk)])
    
    return wk, pi
    
# Wasserstein distance
def w2_distance(x, y, u = None, v = None):
    m, n = len(x), len(y)
    d, _d = x.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    if u is None:
        u = np.ones(m) / m
    if v is None:
        v = np.ones(n) / n
    C = distance_matrix(x, y)
    # C = cost_matrix(x, y)
    pi = sinkhorn.sinkhorn_pi(C, v, u)
    dist = (C * pi).sum()
    return dist, pi

# clustering indices report
def report_indices(kc, wkc, y, true_c = None, title = None, point_winner = True):
    
    # external indices
    print()
    if not title is None:
        n, d = y.shape
        print(title)
        print()
        print(f'dataset size:       {n}')
        print(f'number of features: {d}')
        if not true_c is None:
            print(f'number of classes:  {len(set(true_c))}')
    
    if not true_c is None:
        print()
        print('External indices')
        print()
        
        # Rand
        k_index  = metrics.rand_score(true_c, kc)
        wk_index = metrics.rand_score(true_c, wkc)
        point = [' ', ' '] if not point_winner or np.isclose(k_index, wk_index) else ['*', ' '] if k_index > wk_index else [' ', '*']
        print(f'Rand            \tk-means {k_index:7.4f}' + point[0] + f'   weak k-means {wk_index:6.4f}' + point[1])
        
        # Fowlkes–Mallows
        k_index  = metrics.fowlkes_mallows_score(true_c, kc)
        wk_index = metrics.fowlkes_mallows_score(true_c, wkc)
        point = [' ', ' '] if not point_winner or np.isclose(k_index, wk_index) else ['*', ' '] if k_index > wk_index else [' ', '*']
        print(f'Fowlkes–Mallows \tk-means {k_index:7.4f}' + point[0] + f'   weak k-means {wk_index:6.4f}' + point[1])
        
        
        # homogeneity
        k_index  = metrics.homogeneity_score(true_c, kc)
        wk_index = metrics.homogeneity_score(true_c, wkc)
        point = [' ', ' '] if not point_winner or np.isclose(k_index, wk_index) else ['*', ' '] if k_index > wk_index else [' ', '*']
        print(f'Homogeneity     \tk-means {k_index:7.4f}' + point[0] + f'   weak k-means {wk_index:6.4f}' + point[1])
    
    # internal indices
    print()
    print('Internal indices')
    print()
    
    # Davies-Bouldin
    k_index = metrics.davies_bouldin_score(y, kc)
    wk_index = metrics.davies_bouldin_score(y, wkc)
    point = [' ', ' '] if not point_winner or np.isclose(k_index, wk_index) else ['*', ' '] if k_index < wk_index else [' ', '*']   # inverted!
    print(f'Davies-Bouldin   \tk-means {k_index:7.4f}' + point[0] + f'   weak k-means {wk_index:6.4f}' + point[1])

    # silhouette
    if len(y) <= 10000:
        k_index = metrics.silhouette_score(y, kc)
        wk_index = metrics.silhouette_score(y, wkc)
        point = [' ', ' '] if not point_winner or np.isclose(k_index, wk_index) else ['*', ' '] if k_index > wk_index else [' ', '*']
        print(f'Silhouette       \tk-means {k_index:7.4f}' + point[0] + f'   weak k-means {wk_index:6.4f}' + point[1])


# --- auxiliary internal functions ---

def exx(x, u = None):
    if u is None:
        u = np.repeat(1 / len(x), len(x))
    return sum(u * (x ** 2).sum(axis=1))
         
def exy(x, y, pi):
    m, d  = x.shape
    n, _d = y.shape
    assert _d == d
    assert pi.shape == (m, n)
    return sum(sum(pi * sum([np.outer(x[:,i], y[:,i]) for i in range(d)])))           # equivalent

def random_pi(m, n, v):
    pi = np.random.random([m, n])
    pi = pi / pi.sum(axis=0)
    pi = pi * v
    return pi
    
# Cij = |xi - yj| ** 2
def distance_matrix(x, y):
    m, n = len(x), len(y)
    d, _d = x.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    
    # squared distance matrix
    C = np.zeros([m, n])
    for i in range(d):
        x_, y_ = x[:,i], y[:,i]
        xx = np.repeat(x_, n).reshape([m, n])
        yy = np.repeat(y_, m).reshape([n, m]).T
        C = C + (xx - yy) ** 2
    return C

# Cij = xi * yj
def crossprod_matrix(x, y):
    m, n = len(x), len(y)
    d, _d = x.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    
    # cross product matrix
    C = np.zeros([m, n])
    for i in range(d):
        x_, y_ = x[:,i], y[:,i]
        xx = np.repeat(x_, n).reshape([m, n])
        yy = np.repeat(y_, m).reshape([n, m]).T
        C = C - xx * yy
    # (compare) q = -np.array([sum(w * (xh[i, :] * y[j, :]).sum(axis=0)) for i in range(m) for j in range(n)])
    return C