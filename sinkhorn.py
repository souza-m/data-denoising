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

def newton_method(func, func_prime, x0, tol=1e-6, max_iter=100):
    x = x0

    for iterations in range(max_iter):
        f_x = func(x)
        f_prime_x = func_prime(x)

        if abs(f_prime_x) < 1e-5:
            print("Derivative is close to zero. Newton's method may not converge.")
            return x

        x -= f_x / f_prime_x

        if np.isnan(x).any():
            print("Error: x Calculation returned NaN")
            sys.exit(1)
        elif np.isnan(f_x).any():
            print("Error: f_x Calculation returned NaN")
            sys.exit(1)
        elif np.isnan(f_prime_x).any():
            print("Error: f_prime_x Calculation returned NaN")
            sys.exit(1)

        if abs(f_x) < tol:
            return x

    print("Newton's method did not converge after {} iterations.".format(max_iter))
    print(f'The value of x is {x}')
    return x

def sinkhorn_knopp_(eta, nu, x, y, num_iters=1000, stopping_criterion=1e-9):
    ''' Adapted algorithm with free x-marginals'''
    """
    Sinkhorn Knopp Algorithm to solve Optimal Transport problem

    Parameters:
    eta : float
        Regularization parameter
    nu : np.ndarray, the weight of the y marginal supporting on m points, shape (m,)
    x : np.ndarray, the points of the x marginal supporting on n points shape (n,2)
    y : np.ndarray, the points of the y marginal supporting on m points shape (m,2)
    num_iters : int, optional
        Number of iterations
    stopping_criterion : target stepwise change, float, optional

    Returns:
    u : np.ndarray, the dual variable corresponding to the second moment constraint, shape (1,)
    v : np.ndarray, the dual variable corresponding to the y marginal constraint shape (m,)
    P : np.ndarray, the optimal primal joint distribution pi, shape (n, m)
        Optimal transport plan
    """
    n = x.shape[0]
    m = nu.shape
    mu = np.ones(n) / n # we assume that the true mu is supporting on that n points
    npn = np.newaxis
    xx = np.sum(x ** 2, axis=1)
    xxxx = np.sum(x ** 4, axis=1)
    x_1 = x[:, 0]
    xx_1 = x[:, 0] ** 2
    x_2 = x[:, 1]
    xx_2 = x[:, 1] ** 2

    # Initialization
    v = np.ones(m) / m
    u = 1
    w_1 = 1
    w_2 = 1

    # creating the cost matrix C where the (i,j) th element is x_i dot product y_j
    C = -np.einsum('ij,kj->ik', x, y)
    
    for i in range(num_iters):
        # print(f'Iteration {i}')
        # store previous values
        v_prev = v
        u_prev = u
        w_1_prev = w_1
        w_2_prev = w_2

        # Update v and u based on the formula in section 4.1 in the note "Sinkhorn.tex" insider the folder "Joshua material" on Overleaf
        # Update v

        v = - eta * np.log(np.sum(np.exp((w_1 * x_1[:, npn] + w_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * mu[:, npn], axis=0))

        # Update u by solving a Newton's method
        def func_u(uu):
                f = 1 - np.sum(np.exp((v[npn, :] + w_1 * x_1[:, npn] + w_2 * x_2[:, npn] + uu * xx[:, npn] - C) / eta) * xx[:, npn] * mu[:, npn] * nu[npn, :])
                return f
            
        def grad_func_u(uu):
                f = - np.sum(np.exp((v[npn, :] + w_1 * x_1[:, npn] + w_2 * x_2[:, npn] + uu * xx[:, npn] - C) / eta) * xxxx[:, npn] * mu[:, npn] * nu[npn, :]) / eta
                return f
            
        x0 = 0
        u = newton_method(func_u, grad_func_u, x0, tol=1e-6, max_iter=100)

        # add centered-x constraint
        # Update w_1 by solving a Newton's method
        def func_w1(ww_1):
                f = - np.sum(np.exp((v[npn, :] + ww_1 * x_1[:, npn] + w_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * x_1[:, npn] * mu[:, npn] * nu[npn, :])
                return f
            
        def grad_func_w1(ww_1):
                f = - np.sum(np.exp((v[npn, :] + ww_1 * x_1[:, npn] + w_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * xx_1[:, npn] * mu[:, npn] * nu[npn, :]) / eta
                return f
            
        x0 = 0
        w_1 = newton_method(func_w1, grad_func_w1, x0, tol=1e-6, max_iter=100)

        # Update w_2 by solving a Newton's method
        def func_w2(ww_2):
                f = - np.sum(np.exp((v[npn, :] + w_1 * x_1[:, npn] + ww_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * x_2[:, npn] * mu[:, npn] * nu[npn, :])
                return f
            
        def grad_func_w2(ww_2):
                f = - np.sum(np.exp((v[npn, :] + w_1 * x_1[:, npn] + ww_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * xx_2[:, npn] * mu[:, npn] * nu[npn, :]) / eta
                return f
            
        x0 = 0
        w_2 = newton_method(func_w2, grad_func_w2, x0, tol=1e-6, max_iter=100)

        if num_iters % 10 == 0:
            # calculate the error
            error = np.sum(np.abs(u - u_prev)) + np.sum(np.abs(v - v_prev)) + np.sum(np.abs(w_1 - w_1_prev)) + np.sum(np.abs(w_2 - w_2_prev))

            if error < stopping_criterion:
                # if the error is smaller than the stopping criterion, break the loop
                print(f'The number of iteration it use is {num_iters}')
                break

    # Compute the optimal distribution
    P = np.exp((v[npn, :] + w_1 * x_1[:, npn] + w_2 * x_2[:, npn] + u * xx[:, npn] - C) / eta) * mu[:, npn] * nu[npn, :]

    # Return the dual variables and the optimal distribution
    return u, v, P

