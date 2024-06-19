# -*- coding: utf-8 -*-

import numpy as np
import sys 

# optimize pi with both (y- and x-) maginals fixed, OT algorithm
# uses Sinkhorn algorithm to optimize a matrix of weights
# see Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NIPS)
# C (m x n) is the distance/cost matrix
# lamb is a dual penalization parameter - see theory
# pi (m x n) is the desired matrix of weights (output)
# def sinkhorn_pi(C, v, u, lamb = 100):
#     m, n = C.shape
#     v = v.reshape(n, 1)       # to matrix n x 1
#     u = u.reshape(m, 1)       # to matrix m x 1
#     _C = C / C.max()          # workaround - the algorithm is breaking for larger M's
#     K = np.exp(-lamb * _C)
#     U = K * _C
#     D, L, u, v = sinkhornTransport(u, v, K, U, lamb, verbose = 0)
#     pi = (K.T * u[:,0]).T * v[:,0]
#     return pi

def navieSinkhorn(eta, mu, nu, C, dim_mu, dim_nu, num_iterations=1000, stopping_criterion=1e-9):
    npn = np.newaxis
    mu = np.squeeze(mu)
    nu = np.squeeze(nu)
    u = np.ones(dim_mu) / dim_mu
    v = np.ones(dim_nu) / dim_nu

    for iteration in range(num_iterations):
        #if iteration % 100 == 0:
            #print(f'The number of iteration is {iteration}')

        u_prev = u
        v_prev = v

        u = -eta * np.log(np.sum(np.exp((v[npn, :] - C) / eta) * nu[npn, :], axis=1))
        v = -eta * np.log(np.sum(np.exp((u[:, npn] - C) / eta) * mu[:, npn], axis=0))
        
        if iteration % 1 == 0:
            error = np.sum(np.abs(u - u_prev)) + np.sum(np.abs(v - v_prev))
            #if iteration % 100 == 0:
                #print(f'The error at {iteration} th iteration is {error}')
            if error < stopping_criterion:
                print(f'number of iterations {iteration}')
                break
    
    P = np.exp((u[:, npn] + v[npn, :] - C) / eta) * mu[:, npn] * nu[npn, :]

    return P

# define a new sinkhorn_pi_2 function that use our own sinkhorn algorithm
def sinkhorn_pi(C, v, u, eta = .01, max_iters = 100):
    m, n = C.shape
    v = v.reshape(n, 1)       # to matrix n x 1
    u = u.reshape(m, 1)       # to matrix m x 1
    pi = navieSinkhorn(eta, u, v, C, m, n)
    return pi
    tol = 1e-8

    """
    Sinkhorn-Knopp Algorithm to solve Optimal Transport problem

    Parameters:
    C: Cost matrix
    r: Source distribution (row sums of the optimal transport plan)
    c: Target distribution (column sums of the optimal transport plan)
    lam: Regularization parameter
    max_iters: Maximum number of iterations
    tol: Tolerance for convergence

    Returns:
    P: Optimal transport plan
    """

    # Initialize _u and _v
    _u = np.ones_like(u)
    _v = np.ones_like(v)

    # Sinkhorn iterations
    K = np.exp(- C / eta)
    print(f'K shape is {K.shape}')
    for _ in range(max_iters):
        u_prev, v_prev = _u, _v
        _u = u / np.dot(K, _v)
        _v = u / np.dot(K.T, _u)
        if np.allclose(_u, u_prev, atol=tol) and np.allclose(_v, v_prev, atol=tol):
            break
    print(f'_u shape is {_u.shape}')
    print(f'_v shape is {_v.T.shape}')
    # Compute optimal transport plan P
    P = u * K * v.T

    return P