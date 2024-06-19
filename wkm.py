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
    _y = y - barycenter
    x = None if x0 is None else x0 - barycenter
    obj_series = []
    
    # principal curve with bounded curvature
    if method == 'curve':
 
        # parameters
        curve_penalty = kwargs.get('curve_penalty', 0)
        alpha = .01
        if x0 is None or pi0 is None:
            raise ValueError("initial position and transport plan must be provided")
        if exx(x0) == 0:
            raise ValueError("initial position must have positive variance")
            
        # initial position and transport plan
        pi = pi0
        xh = x / np.sqrt(exx(x))
        x = xh * exy(xh, _y, pi)   # rescale
        
        # iterate
        for epoch in range(epochs):
            xh = update_xh(pi, _y, 'curve', xh, curve_penalty, alpha)
            pi = update_pi(xh, _y, 'nearest', pi0)
            x = xh * exy(xh, _y, pi)   # rescale
            
            # report
            EXX = exx(x, pi.sum(axis=1))
            EYY = exx(_y)
            EXY = exy(x, _y, pi)    
            obj_series.append(EXY)
            if verbose and (epochs <= 20 or epoch+1 <= 5 or epoch+1 == 10 or (epoch+1)%50 == 0):
                r = EXY / np.sqrt(EXX * EYY)
                R2 = r ** 2
                zero_weights = np.isclose(pi.sum(axis=1), 0, atol=1e-5)
                print(f'{epoch+1:4d}     R2 = {R2:6.2%}   # nonzero weights = {m - zero_weights.sum():3d} / {m}')
    
    # k-means with fixed weights
    elif method == 'fixed_u':
        
        # parameter
        u = kwargs.get('u', np.ones(m) / m)
        
        # initial position
        if x0 is None:
            # form initial pi
            pi = pi0 if not pi0 is None else random_pi(m, n, v=v)
            xh = update_xh(pi, _y, 'direct')
        else:
            # given
            x = x0
            xh = x / np.sqrt(exx(x))
        
        # iterate
        for epoch in range(epochs):
            pi = update_pi(xh, _y, 'sinkhorn_fixed_u', u=u, v=v)
            xh = update_xh(pi, _y, 'direct')
            x = xh * exy(xh, _y, pi)   # rescale
            
            # report
            EXX = exx(x, pi.sum(axis=1))
            EYY = exx(_y)
            EXY = exy(x, _y, pi)    
            obj_series.append(EXY)
            if verbose and (epochs <= 20 or epoch+1 <= 5 or epoch+1 == 10 or (epoch+1)%50 == 0):
                r = EXY / np.sqrt(EXX * EYY)
                R2 = r ** 2
                zero_weights = np.isclose(pi.sum(axis=1), 0, atol=1e-5)
                print(f'{epoch+1:4d}     R2 = {R2:6.2%}   # nonzero weights = {m - zero_weights.sum():3d} / {m}')
    
    else:
        print('--- not implemented ---')
    
    # wrap and return
    if not x is None:
        x += barycenter
    return x, pi, obj_series


# --- optimization functions ---

# update xh
xh_methods = ['direct', 'curve']
def update_xh(pi, y, method, xh0 = None, curve_penalty = 0, alpha = 1):
    m, n = pi.shape
    _, d = y.shape
    u = pi.sum(axis=1)
    
    # find new xh
    yhat = np.dot(pi, y)
    if method == 'direct' or (method == 'curve' and curve_penalty == 0):
        _xh = ellipse_max(yhat, u, centered=True)
    elif method == 'curve':
        if xh0 is None:
            print('--- error: xh0 must be given')
        penalty = curve_penalty * phi(xh0)
        _xh = ellipse_max(yhat - penalty, u)
    else:
        print('--- error: method not identified')
        return
        
    # check barycenter constraint
    xh_bary = (u[:,None] * _xh).sum(axis=0)
    assert np.isclose(xh_bary, np.zeros(d), atol=1e-6).all(), 'barycenter condition violated'
        
    # check second moment constraint
    xh_secmon = sum((u[:,None] * _xh ** 2).sum(axis=0))
    assert np.isclose(xh_secmon, 1, atol=1e-6), 'second moment condition violated'
    if xh_secmon < 1 - 1e-6:
        print('--- warning: variance less than 1', xh_secmon)
    
    # partial update and return
    if alpha == 1 or xh0 is None:
        xh = _xh
    else:
        xh = alpha * _xh + (1. - alpha) * xh0
        xh /= np.sqrt(exx(xh, pi.sum(axis=1)))
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
    D = [np.zeros([1,d])] + [-E[i] * (1 / anorm[i+1] + 1 / anorm[i]) for i in range(len(E))] + [np.zeros([1,d])]
    cpenalty = np.vstack(D)
    return cpenalty

# call appropriate method to optimize pi
# check constraints
# update (partially if alpha < 1)
pi_methods = ['lp_free_u', 'sinkhorn_free_u', 'sinkhorn_fixed_u', 'nearest']
def update_pi(xh, y, method, pi0 = None, u = None, v = None, alpha = 1):
    d, _d = xh.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    
    if method == 'nearest':
        if pi0 is None:
            print('--- error: pi0 must be given')
            return
        C = distance_matrix(xh, y)
        _pi = nearest_pi(C, pi0)
    elif method == 'sinkhorn_fixed_u':
        # rescale y in order to use a common entropy penalty parameter to Sinkhorn
        # notice that xh is already normalized
        _y = y / np.sqrt(exx(y))
        C = crossprod_matrix(xh, _y)
        _pi = sinkhorn.sinkhorn_pi(C, v, u)
    elif method == 'sinkhorn_free_u':
        # v...
        # C = crossprod_matrix(xh, y)
        # xh_sq = np.sum(xh*xh, axis=1)
        # _pi = np.maximum(_pi, 0.)
        # _pi = _pi * v[None,:] / _pi.sum(axis=0)[None,:]
        print('--- not implemented ---')
        return
    elif method == 'lp_free_u':
        # v...
        # C = crossprod_matrix(xh, y)
        # xh_sq = np.sum(xh*xh, axis=1)
        # _pi = lp_pi(C, v, xh, xh_sq, floor = 0, ceil = 1)
        print('--- not implemented ---')
        return
    else:
        print('methods: sinkhorn_fixed, sinkhorn_free, lp_free')
        return
    
    # update and return
    pi = _pi if pi0 is None else alpha  * _pi + (1. - alpha) * pi0
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
    if len(sigma) == 1:
        return sigma
    m = len(sigma)
    # swap first assignment?
    for i in range(1, m):
        if C[0, sigma[i]] + C[i, sigma[0]] < C[0, sigma[0]] + C[i, sigma[i]]:
            # print('--- swap', sigma[0], sigma[i])
            sigma[0], sigma[i] = sigma[i], sigma[0]
            return best_assignment(C, sigma)
    return [sigma[0]] + best_assignment(C[1:,:], sigma[1:])

# find exact optimal pi
# pi0 must be composed of 0's and (1/n)'s
# NOTE: still not exact
def nearest_pi(C, pi0):
    _sigma = pi_to_sigma(pi0)
    sigma = best_assignment(C, _sigma)
    count = 0
    while count < len(pi0) and not sigma == _sigma:
        count += 1
        if count > 5:
            print('--- note: nearest_pi reiteration', count)
        _sigma , sigma = sigma, best_assignment(C, sigma)
    return sigma_to_pi(sigma)


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