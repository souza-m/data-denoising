# -*- coding: utf-8 -*-
"""
Created on Sep-2023

@author: souza-m
"""

import numpy as np
from scipy.optimize import linprog as lp
import sinkhorn
from sklearn import metrics

'''
notation:
    y  (n,d) is the noisy data (input)
    x  (m,d) is the denoised set of points (solution)
    xh (m,d) is the auxiliary variable in the unit ellipse
    u is the x-marginal probability distribution
    v is the y-marginal probability distribution
    pi (m,n) is the transport matrix
'''


# --- main function ---

fit_methods = ['kmeans_fixed', 'kmeans_variable', 'curvature', 'length']
def fit(y, m, method, x0 = None, pi0 = None, epochs = 1, verbose = False, **kwargs):
    
    # initialize and centralize
    n = len(y)
    u = kwargs.get('u', np.ones(m) / m)
    v = kwargs.get('v', np.ones(n) / n)
    barycenter = np.dot(v, y)
    _y = y - barycenter
    x = None if x0 is None else x0 - barycenter
    obj_series = []
    
    # principal curve with bounded curvature
    if method == 'curvature':

        # parameters
        curvature_penalty = kwargs.get('curvature_penalty', 0)
        alpha = .0001
        if x0 is None or pi0 is None:
            raise ValueError('initial position and transport plan must be provided')
        if exx(x0) == 0:
            raise ValueError('initial position must have positive variance')
            
        # initial position and transport plan
        pi = pi0
        xh = x / np.sqrt(exx(x))
        x = xh * exy(xh, _y, pi)   # rescale
        
        # iterate
        for epoch in range(epochs):
            _xh = update_xh(pi, _y, method='curvature', xh0=xh, curvature_penalty=curvature_penalty, alpha=alpha)
            _xh = update_xh(pi, _y, 'curvature', xh0 = xh, curvature_penalty = curvature_penalty, alpha = alpha)
            if epoch % 1000 == 0:
                d = _xh.shape[1]
                dif = max(max(np.abs(xh[i,k] - _xh[i,k]) for k in range(d)) for i in range(m))
                print(dif)
                if dif < 1e-5:
                    print('convergence achieved')
                    break
            xh = _xh
            x = xh * exy(xh, _y, pi)   # rescale
            if epoch % 10 == 0:
                if False:# m == n:
                    pi = update_pi(xh, _y, 'nearest', pi0)
                else:
                    pi = update_pi(xh, _y, 'sinkhorn_fixed', pi0, u, v)
            
            # report
            EXX = exx(x, pi.sum(axis=1))
            EYY = exx(_y)
            EXY = exy(x, _y, pi)    
            obj_series.append(EXY)
            if verbose and (epochs <= 20 or epoch+1 <= 5 or epoch+1 == 10 or (epoch+1)%100 == 0):
                r = EXY / np.sqrt(EXX * EYY)
                R2 = r ** 2
                zero_weights = np.isclose(pi.sum(axis=1), 0, atol=1e-5)
                print(f'{epoch+1:4d}     R2 = {R2:6.2%}   # nonzero weights = {m - zero_weights.sum():3d} / {m}')
    
    # principal curve with bounded length
    elif method == 'length':
        
        # bound
        B = kwargs.get('length', 0)
        # _B = 1.
        print('B', B)
        
        # initial position
        if x0 is None:
            # form some random pi
            pi = pi0 if not pi0 is None else random_pi(m, n, v=v)
            # xh = update_xh(pi, _y, 'length', B=B)
            xh = update_xh(pi, _y, 'length', B=B)
        else:
            # given
            x = x0
            xh = x / np.sqrt(exx(x))
        
        # iterate
        for epoch in range(epochs):
            # _B = 1. - (1. - B) * (epoch + 1) / epochs
            pi = update_pi(xh, _y, 'sinkhorn_fixed', u=u, v=v)
            xh = update_xh(pi, _y, 'length', B=B)
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

    # k-means
    elif method in ['kmeans_fixed', 'kmeans_variable']:
        
        # initial position
        if x0 is None:
            # form some random pi
            pi = pi0 if not pi0 is None else random_pi(m, n, v=v)
            xh = update_xh(pi, _y, 'direct')
        else:
            # given
            x = x0
            xh = x / np.sqrt(exx(x))
        
        # iterate
        for epoch in range(epochs):
            if method == 'kmeans_fixed':
                pi = update_pi(xh, _y, 'sinkhorn_fixed', u=u, v=v)
            else:
                pi = update_pi(xh, _y, 'lp_free_u', v=v)
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
xh_methods = ['direct', 'curvature', 'length']
def update_xh(pi, y, method, xh0 = None, curvature_penalty = 0, B = 0, alpha = 1):
    m, n = pi.shape
    _, d = y.shape
    u = pi.sum(axis=1)
    
    # find new xh
    yhat = np.dot(pi, y)
    if method == 'direct' or (method == 'curvature' and curvature_penalty == 0):
        _xh = ellipse_max(yhat, u, centered=True)
    elif method == 'curvature':
        if xh0 is None:
            print('--- error: xh0 must be given')
        penalty = curvature_penalty * phi(xh0)  # linear penalization based on previous xh
        _xh = ellipse_max(yhat - penalty, u)
    elif method == 'length':
        print('update_xh: B = ', B)
        _xh = qp_length(yhat, B)
        xh_secmom = sum((u[:,None] * _xh ** 2).sum(axis=0))
        assert xh_secmom < 1 + 1e-6, 'variance greater than 1'
        _xh /= np.sqrt(xh_secmom)
    else:
        print('--- error: method not identified')
        return
        
    # check barycenter constraint
    xh_bary = (u[:,None] * _xh).sum(axis=0)
    # assert np.isclose(xh_bary, np.zeros(d), atol=1e-4).all(), f'barycenter condition violated {xh_bary}'
    if not np.isclose(xh_bary, np.zeros(d), atol=1e-4).all():
        print(f'barycenter condition violated {xh_bary}')
        
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

# max C.x
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

from cvxopt import solvers, matrix

def qp_length(C, B):
    print('qp_length: B = ', B)
    # https://cvxopt.org/userguide/coneprog.html?highlight=qp#second-order-cone-programming
    # max C.x
    # st  u.x   == 0
    #     u.x^2 <= 1
    #     sum || xi - xi-1 || == B
    m, d = C.shape
    
    # objective (to minimize)
    c = matrix(np.concatenate([-C.reshape(d * m), np.zeros(m - 1)]))
    
    # centered x: sum{ xi^l == 0 } for l = 1, ..., d
    A1 = np.hstack([np.eye(d) for i in range(m)] + [np.zeros([d, m - 1])])
    b1 = np.zeros(d)
    
    # bounded length: sum{ ai } == B
    A2 = np.hstack([np.zeros(d * m), np.ones(m - 1)])
    b2 = B * np.ones(1)
    
    # equality constraints
    A = np.vstack([A1, A2])
    b = np.concatenate([b1, b2])
    A = matrix(A)
    b = matrix(b)
    
    Gq = []
    hq = []
    
    # std(x) <= 1
    G = np.vstack([np.zeros(d * m + m - 1), np.hstack([-np.eye(d * m), np.zeros([d * m, m - 1])])])
    h = np.concatenate([np.ones(1), np.zeros(d * m)])
    Gq.append(matrix(G))
    hq.append(matrix(h))
    
    # bounded length: sum || xi+1 - xi || <= ai
    for i in range(m - 1):
        G0 = -np.eye(1, d * m + m - 1, k = d * m + i)
        G1 = np.hstack(i * [np.zeros([d, d])] + [-np.eye(d)] + [np.eye(d)] + (m - 2 - i) * [np.zeros([d, d])] + [np.zeros([d, m - 1])])
        # print(G1)
        G = np.vstack([G0, G1])
        h = np.zeros(d + 1)
        Gq.append(matrix(G))
        hq.append(matrix(h))
    
    sol = solvers.socp(c, A=A, b=b, Gq=Gq, hq=hq)
    xh = np.array(sol['x'])[:d * m].reshape([m, d])
    
    length = np.array(sol['x'])[d * m]
    print('B', B, 'length', length)
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
pi_methods = ['lp_free_u', 'sinkhorn_fixed', 'nearest']
def update_pi(xh, y, method, pi0 = None, u = None, v = None, alpha = 1):
    d, _d = xh.shape[1], y.shape[1]
    assert d == _d, 'dimension mismatch'
    
    if method == 'nearest':
        if pi0 is None:
            print('--- error: pi0 must be given')
            return
        C = distance_matrix(xh, y)
        _pi = nearest_pi(C, pi0)
    elif method == 'sinkhorn_fixed':
        # rescale y in order to use a common entropy penalty parameter to Sinkhorn
        # notice that xh is already normalized
        _y = y / np.sqrt(exx(y))
        C = crossprod_matrix(xh, _y)
        _pi = sinkhorn.sinkhorn_pi(C, u, v)
    elif method == 'lp_free_u':
        C = crossprod_matrix(xh, y)
        xh_sq = np.sum(xh**2, axis=1)
        _pi = lp_pi(C, v, xh, xh_sq)
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

# optimize pi with y-maginal fixed, linear program (or quadratic if pi_ij is penalized)
# xh_sq is the vector of |xi|**2
# pi is bounded as
#    sum_i sum_j pi_ij * xhi = 0
#    sum_i sum_j pi_ij * xh_sq_i <= 1
def lp_pi(C, v, xh, xh_sq, floor = 0, ceil = 1):
    m, n = C.shape
    d = xh.shape[1]
    
    q = C.copy()
    q.resize(m * n)
    
    # y-marginal constraint
    A1 = np.vstack([np.tile(np.hstack([np.zeros(i), np.ones(1), np.zeros((n-i-1))]), m) for i in range(n)])
    b1 = v
    
    # centered_x constraint
    Z = [np.repeat(xh[:,k], n) for k in range(d)]
    A2 = np.vstack(Z)
    b2 = np.zeros(d)
    
    # second moment as an equality
    A3 = np.repeat(xh_sq, n)
    A3.resize(1, len(A3))
    b3 = np.ones(1)
    
    A = np.vstack([A1, A2, A3])
    b = np.concatenate([b1, b2, b3])

    # format:
    #  min  q' pi
    #  st   A_ub pi <= b_ub
    #       A_eq pi == b_eq
    lp_resut = lp(c=q, A_eq=A, b_eq=b)
    pi = lp_resut['x'].reshape((m, n))
        
    return pi


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