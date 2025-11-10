from cvxopt import matrix, solvers, spmatrix, sparse
import numpy as np

def find_coupling_cvxopt(x, y, nu, la, no_clusters, no_data):
    """
    CVXOPT version of your QP.

    Args:
        x: (n, d) cluster centers (or x_i)
        y: (k, d) data points (or y_j)
        nu: (k,) nonnegative weights with sum over i of gamma_{i,j} == nu[j]
        la: scalar lambda >= 0
        no_clusters: n
        no_data: k
        c: a callable c(X, Y) returning the (n, k) cost matrix used for total_value (same as your original)
    Returns:
        ga_opt: (n, k) optimal gamma
        total_value: float, sum(ga_opt * C) using the provided c
    """
    solvers.options['show_progress'] = False  # quiet

    n, k = no_clusters, no_data
    d = y.shape[1]
    assert x.shape == (n, d)
    assert y.shape == (k, d)
    nu = np.asarray(nu, dtype=float).reshape(k,)
    assert nu.shape == (k,)
    assert la >= 0

    # Precalculate distances used in linear part (same as your Gurobi code)
    dist_squared = np.sum((x[:, None, :] - y[None, :, :])**2, axis=2)  # (n,k)

    # Build QP matrices: 1/2 z^T P z + q^T z
    # z = vec(ga) with index idx(i,j) = i*k + j
    # Quadratic part: la * sum_i ||Y^T g_i - x_i||^2
    #    = la * [ g_i^T (Y Y^T) g_i - 2 (Y x_i)^T g_i + ||x_i||^2 ]
    # In QP form:
    #    P block for each i: 2*la*(Y Y^T) (since CVXOPT uses 1/2 z^T P z)
    #    q block for each i: dist_block - 2*la*(Y x_i)
    Y = y  # (k, d)
    YYT = Y @ Y.T  # (k, k), PSD
    P_block = 2.0 * la * YYT  # (k,k)

    # Build P as block-diagonal: kron(I_n, P_block)
    if la == 0:
        P = np.zeros((n*k, n*k))
    else:
        P = np.kron(np.eye(n), P_block)

    # Linear term q
    q = dist_squared.reshape(n*k)
    if la != 0:
        for i in range(n):
            q_i = -2.0 * la * (Y @ x[i])  # (k,)
            q[i*k:(i+1)*k] += q_i

    # Inequality constraints: ga >= 0  ->  -I z <= 0
    G = -np.eye(n*k)
    h = np.zeros(n*k)

    # Equality constraints: sum_i ga[i,j] = nu[j] for each j
    # Row j has 1 at positions i*k + j for all i
    A = np.zeros((k, n*k))
    for j in range(k):
        for i in range(n):
            A[j, i*k + j] = 1.0
    b = nu.copy()

    # Convert to cvxopt matrices
    P_cvx = matrix(P)
    q_cvx = matrix(q)
    G_cvx = matrix(G)
    h_cvx = matrix(h)
    A_cvx = matrix(A)
    b_cvx = matrix(b)

    # Solve
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)

    z = np.array(sol['x']).reshape(n*k)
    ga_opt = z.reshape(n, k)

    def d2(x, y):
        return np.sum((x - y) ** 2, axis=-1)

    # Compute total_value with your provided c(X,Y)
    X = x[:, None, :]     # (n,1,d)
    Yb = y[None, :, :]    # (1,k,d)
    C = d2(X, Yb)         # (n,k)
    total_value = float(np.sum(ga_opt * C))

    return ga_opt, total_value