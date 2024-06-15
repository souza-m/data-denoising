# -*- coding: utf-8 -*-

import numpy as np
import sys 


def sinkhornTransport(a, b, K, U, lamb, stoppingCriterion='marginalDifference', p_norm=np.inf, tolerance=.5e-2,
                      max_iter=5000, verbose=0):
    '''
    This Code is Python translation of Cuturi's Optimal Transport Algorithm, original MATLAB code can be found at:
    https://marcocuturi.net/SI.html

    Original Paper discussing the Algorithm can be found at:
    
    https://papers.nips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    

    This Code is Translated by Hau Phan.

    inputs: 

    a is either a n x 1 column vector in the probability simplex (nonnegative, summing to one). This is the [1-vs-N mode]
    - a n x N matrix, where each column vector is in the probability simplex. This is the [N x 1-vs-1 mode]
    
    b is a m x N matrix of N vectors in the probability simplex.
    
    K is a n x m matrix, equal to exp(-lambda M), where M is the n x m matrix of pairwise distances between bins 
    described in a and bins in the b_1,...b_N histograms. In the most simple case n = m and M is simply a distance matrix (zero
    on the diagonal and such that m_ij < m_ik + m_kj

    U = K.*M is a n x m matrix, pre-stored to speed up the computation of the distances.

    Optional Inputs:

    stoppingCriterion in {'marginalDifference','distanceRelativeDecrease'}
    - marginalDifference (Default) : checks whether the difference between 
    the marginals of the current optimal transport and the theoretical marginals set by a b_1,...,b_N are satisfied.
    - distanceRelativeDecrease : only focus on convergence of the vector of distances

    p_norm: parameter in {(1,+infty]} used to compute a stoppingCriterion statistic
    rom N numbers (these N numbers might be the 1-norm of marginal
    differences or the vector of distances.

    tolerance : > 0 number to test the stoppingCriterion.

    maxIter: maximal number of Sinkhorn fixed point iterations.
    
    verbose: verbose level. 0 by default.
    
    Output

    D : vector of N dual-sinkhorn divergences, or upper bounds to the Eearth Movers Disatnce.

    L : vector of N lower bounds to the original OT problem, a.k.a EMD. This is computed by using
    the dual variables of the smoothed problem, which, when modified
    adequately, are feasible for the original (non-smoothed) OT dual problem

    u : n x N matrix of left scalings
    v : m x N matrix of right scalings

    The smoothed optimal transport between (a_i,b_i) can be recovered as
    T_i = np.diag(u[:,i]) @ K @ diag(v[:,i]);

    '''

    if a.shape[1] == 1:
        one_vs_n = True
    elif a.shape[1] == b.shape[1]:
        one_vs_n = False
    else:
        print(
            "The first parameter a is either a column vector in the probability simplex, or N column vectors in the probability simplex where N is size(b,2)")
        return

    if b.shape[1] > b.shape[0]:
        bign = True
    else:
        bign = False

    if one_vs_n:
        I = np.array(a > 0)
        some_zero_values = False
        if not (np.sum(I) == len(I)):
            some_zero_values = True
            K = K[I.squeeze()]
            U = U[I.squeeze()]
            a = a[I.squeeze()]
        ainvK = K / a
    # fixed point counter
    compt = 0
    # initialization of left scaling factors, N column vectors
    u = np.ones((a.shape[0], b.shape[1])) / a.shape[0]

    if stoppingCriterion == 'distanceRelativeDecrease':
        Dold = np.ones((1, b.shape[1]))

    while compt < max_iter:
        if one_vs_n:
            if bign:
                u = 1 / (ainvK @ (b / (K.T @ u)))
            else:
                u = 1 / (ainvK @ (b / (u.T @ K).T))
        else:
            if bign:
                u = a / (K @ (b / (u.T @ K).T))
            else:
                u = a / (K @ (b / (K.T @ u)))
        compt += 1

        if compt % 20 == 1 or compt == max_iter:
            if bign:
                v = b / (K.T @ u)
            else:
                v = b / (u.T @ K).T

            if one_vs_n:
                u = 1 / (ainvK @ v)
            else:
                u = a / (K @ v)

            if stoppingCriterion == 'distanceRelativeDecrease':
                D = np.sum(u * (U @ v), axis=0)
                Criterion = np.linalg.norm(D / Dold - 1, p_norm)
                if Criterion < tolerance or np.isnan(Criterion):
                    break
                Dold = D
            elif stoppingCriterion == 'marginalDifference':
                temp = np.sum(np.abs(v * (K.T @ u) - b), axis=0)
                Criterion = np.linalg.norm(temp, p_norm)
                if Criterion < tolerance or np.isnan(Criterion):
                    break
            else:
                print("Stopping Criterion not recognized")
                return

            compt += 1
            if verbose > 0:
                print('Iteration :', str(compt), ' Criterion: ', str(Criterion))
            if np.sum(np.isnan(Criterion)) > 0:
                print(
                    'NaN values have appeared during the fixed point iteration. This problem appears because of insufficient machine precision when processing computations with a regularization value of lambda that is too high. Try again with a reduced regularization parameter lambda or with a thresholded metric matrix M')

    if stoppingCriterion == "marginalDifference":
        D = np.sum(u * (U @ v), axis=0)

    alpha = np.log(u)
    beta = np.log(v)
    beta[beta == -np.inf] = 0
    if one_vs_n:
        L = (a.T @ alpha + np.sum(b * beta, axis=0)) / lamb
    else:
        alpha[alpha == -np.inf] = 0
        print(a.shape)
        print(alpha.shape)
        L = (np.sum(a * alpha, axis=0) + np.sum(b * beta, axis=0)) / lamb

    if one_vs_n and some_zero_values:
        uu = u
        u = np.zeros((len(I), b.shape[1]))
        u[I.squeeze()] = uu

    return D, L, u, v

# optimize pi with both (y- and x-) maginals fixed, OT algorithm
# uses Sinkhorn algorithm to optimize a matrix of weights
# see Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NIPS)
# C (m x n) is the distance/cost matrix
# lamb is a dual penalization parameter - see theory
# pi (m x n) is the desired matrix of weights (output)
def sinkhorn_pi(C, v, u, lamb = 100):
    m, n = C.shape
    v = v.reshape(n, 1)       # to matrix n x 1
    u = u.reshape(m, 1)       # to matrix m x 1
    _C = C / C.max()          # workaround - the algorithm is breaking for larger M's
    K = np.exp(-lamb * _C)
    U = K * _C
    D, L, u, v = sinkhornTransport(u, v, K, U, lamb, verbose = 0)
    pi = (K.T * u[:,0]).T * v[:,0]
    return pi

# def navieSinkhorn(eta, mu, nu, C, dim_mu, dim_nu, num_iterations=1000, stopping_criterion=1e-9):
#     npn = np.newaxis
#     mu = np.squeeze(mu)
#     nu = np.squeeze(nu)
#     u = np.ones(dim_mu) / dim_mu
#     v = np.ones(dim_nu) / dim_nu

#     for iteration in range(num_iterations):
#         #if iteration % 100 == 0:
#             #print(f'The number of iteration is {iteration}')

#         u_prev = u
#         v_prev = v

#         u = -eta * np.log(np.sum(np.exp((v[npn, :] - C) / eta) * nu[npn, :], axis=1))
#         v = -eta * np.log(np.sum(np.exp((u[:, npn] - C) / eta) * mu[:, npn], axis=0))
        
#         if iteration % 1 == 0:
#             error = np.sum(np.abs(u - u_prev)) + np.sum(np.abs(v - v_prev))
#             #if iteration % 100 == 0:
#                 #print(f'The error at {iteration} th iteration is {error}')
#             if error < stopping_criterion:
#                 print(f'The number of iteration it use is {iteration}')
#                 break
    
#     P = np.exp((u[:, npn] + v[npn, :] - C) / eta) * mu[:, npn] * nu[npn, :]

#     return P

# define a new sinkhorn_pi_2 function that use our own sinkhorn algorithm
# def sinkhorn_pi_(C, v, u, eta = .06, max_iters = 100):
#     m, n = C.shape
#     v = v.reshape(n, 1)       # to matrix n x 1
#     u = u.reshape(m, 1)       # to matrix m x 1
#     pi = navieSinkhorn(eta, u, v, C, m, n)
#     return pi
#     tol = 1e-8

#     """
#     Sinkhorn-Knopp Algorithm to solve Optimal Transport problem

#     Parameters:
#     C: Cost matrix
#     r: Source distribution (row sums of the optimal transport plan)
#     c: Target distribution (column sums of the optimal transport plan)
#     lam: Regularization parameter
#     max_iters: Maximum number of iterations
#     tol: Tolerance for convergence

#     Returns:
#     P: Optimal transport plan
#     """

#     # Initialize _u and _v
#     _u = np.ones_like(u)
#     _v = np.ones_like(v)

#     # Sinkhorn iterations
#     K = np.exp(- C / eta)
#     print(f'K shape is {K.shape}')
#     for _ in range(max_iters):
#         u_prev, v_prev = _u, _v
#         _u = u / np.dot(K, _v)
#         _v = u / np.dot(K.T, _u)
#         if np.allclose(_u, u_prev, atol=tol) and np.allclose(_v, v_prev, atol=tol):
#             break
#     print(f'_u shape is {_u.shape}')
#     print(f'_v shape is {_v.T.shape}')
#     # Compute optimal transport plan P
#     P = u * K * v.T

#     return P

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

