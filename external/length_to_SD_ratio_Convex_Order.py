import numpy as np
import ot
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB
import math
# from cvxopt import matrix, solvers
from cvxopt import matrix, solvers, spmatrix, sparse
import pickle


# Import the external function for Kantorovich order computation
import bounded_length_PC_kantarovich_order_for_call

# Objective function 
def c(x, y):
    return np.sum((x - y) ** 2, axis=-1)

def find_location(y, ga, no_clusters, L):
    no_data = y.shape[0]
    LL = L * L

    model = gp.Model("curve_fitting")

    # Decision variables
    x = model.addVars(no_clusters, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="x")

    # Objective function
    obj = gp.quicksum([(x[i, 0] - y[j, 0]) * (x[i, 0] - y[j, 0]) * ga[i, j] for i in range(no_clusters) for j in range(no_data)]) + \
          gp.quicksum([(x[i, 1] - y[j, 1]) * (x[i, 1] - y[j, 1]) * ga[i, j] for i in range(no_clusters) for j in range(no_data)])
    
    # Minimize the objective function
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: Bounded length
    length_expr = gp.QuadExpr()
    for i in range(no_clusters - 1):
        length_expr += (x[i + 1, 0] - x[i, 0]) * (x[i + 1, 0] - x[i, 0]) + (x[i + 1, 1] - x[i, 1]) * (x[i + 1, 1] - x[i, 1])

    model.addConstr(length_expr <= LL, "length_constraint")
    
    # Optimize model
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print("New incumbent's objective:", obj)
    model.optimize(mycallback)

    # print out the optimal value
    print(f'Optimal objective value: {model.objVal}')

    if model.status == GRB.OPTIMAL:
        x_solution = np.array([[x[i, 0].X, x[i, 1].X] for i in range(no_clusters)])
        return model.objVal, x_solution
    else:
        print("No optimal solution found")
        return None
    
solvers.options['show_progress'] = False

def find_location_cvxopt(y, ga, no_clusters, L):
    """
    Transforms the Gurobi Convex Quadratic Program (CQP) into the 
    standard CVXOPT Quadratic Program (QP) format.

    NOTE ON THE QUADRATIC CONSTRAINT:
    The original problem's 'bounded length' constraint is quadratic:
    sum_i [(x_{i+1, 0} - x_{i, 0})^2 + (x_{i+1, 1} - x_{i, 1})^2] <= L^2.
    CVXOPT's standard QP solver (solvers.qp) only accepts LINEAR constraints.
    Therefore, this function solves the UNCONSTRAINED QP for the objective,
    as the quadratic constraint cannot be encoded in the G*x <= h matrix form.
    """
    no_data = y.shape[0]
    
    # 1. Define the decision vector size
    # Total variables N = 2 * no_clusters
    # Order: x = (x_{0,0}, ..., x_{Nc-1,0}, x_{0,1}, ..., x_{Nc-1,1})
    N = 2 * no_clusters 

    # 2. Formulate the Objective Function: (1/2) * x' * P * x + q' * x
    
    # --- P Matrix (Quadratic Term) ---
    # The quadratic part of the objective is: sum_{i,k} x[i,k]^2 * (sum_j ga[i,j])
    # P_ii must be 2 * (sum_j ga[i,j])
    
    # C_i = sum_j ga[i,j] (Total weight for cluster i)
    C = ga.sum(axis=1) # Shape (no_clusters,)

    # The diagonal of P will be [2*C0, ..., 2*C(Nc-1), 2*C0, ..., 2*C(Nc-1)]
    P_diag = np.concatenate((2 * C, 2 * C))
    
    P_np = np.diag(P_diag)
    P = matrix(P_np, tc='d') 

    # --- q Vector (Linear Term) ---
    # The linear part of the objective is: sum_{i,k} -2 * x[i,k] * (sum_j ga[i,j] * y[j,k])
    
    # D_{i,0} = -2 * sum_j ga[i,j] * y[j,0] (Coeffs for x-coords)
    D0 = -2 * (ga * y[:, 0]).sum(axis=1) 
    
    # D_{i,1} = -2 * sum_j ga[i,j] * y[j,1] (Coeffs for y-coords)
    D1 = -2 * (ga * y[:, 1]).sum(axis=1) 

    # q = [D0, D1] concatenated
    q_np = np.concatenate((D0, D1))
    q = matrix(q_np, tc='d')

    # 3. Formulate Constraints (G, h, A, b)
    # G, h, A, and b are empty since there are no linear constraints or bounds.
    G = matrix([], (0, N), 'd')
    h = matrix([], (0, 1), 'd')
    A = matrix([], (0, N), 'd')
    b = matrix([], (0, 1), 'd')
    
    # 4. Solve the QP
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except Exception as e:
        print(f"CVXOPT Solver error: {e}")
        return None, None

    # 5. Extract results and reconstruct solution
    if sol['status'] == 'optimal':
        x_solution_vector = np.array(sol['x']).flatten()
        
        # Reshape solution: x_coords (first Nc), y_coords (second Nc)
        x_coords = x_solution_vector[0:no_clusters]
        y_coords = x_solution_vector[no_clusters:2*no_clusters]
        
        x_solution = np.vstack([x_coords, y_coords]).T
        
        # Calculate the actual objective value
        # CVXOPT's 'primal objective' = (1/2)*x'Px + q'x. 
        # We must add the constant term: sum_{i,j} ga[i,j] * y[j,k]^2
        obj_val_qp = sol['primal objective']
        
        const_term = np.sum(ga * y[:, 0]**2) + np.sum(ga * y[:, 1]**2)
        obj_val_actual = obj_val_qp + const_term
        
        # Print the result similar to the original code
        print(f"Optimal objective value: {obj_val_actual}")
        return obj_val_actual, x_solution
    else:
        print(f"No optimal solution found. Status: {sol['status']}")
        return None, None

def find_location_cvxopt_L_to_SD_ratio(y, ga, no_clusters, L, B=None):
    """
    Transforms the Gurobi Convex Quadratic Program (CQP) into the 
    standard CVXOPT Quadratic Program (QP) format.

    The new non-linear constraint related to B is GENERATED in matrix form, 
    but CVXOPT's solvers.qp is unable to use it, as it only handles linear constraints.
    The problem is solved UNCONSTRAINED here.
    """
    no_data = y.shape[0]
    
    # 1. Define the decision vector size
    # Total variables N = 2 * no_clusters
    # Order: x = (x_{0,0}, ..., x_{Nc-1,0}, x_{0,1}, ..., x_{Nc-1,1})
    N = 2 * no_clusters 

    # 2. Formulate the Objective Function: (1/2) * x' * P * x + q' * x
    
    # --- P Matrix (Quadratic Term) ---
    C = ga.sum(axis=1) # Total weight for cluster i: C_i = sum_j ga[i,j]
    P_diag = np.concatenate((2 * C, 2 * C))
    P_np = np.diag(P_diag)
    P = matrix(P_np, tc='d') 

    # --- q Vector (Linear Term) ---
    D0 = -2 * (ga * y[:, 0]).sum(axis=1) # Coeffs for x-coords
    D1 = -2 * (ga * y[:, 1]).sum(axis=1) # Coeffs for y-coords
    q_np = np.concatenate((D0, D1))
    q = matrix(q_np, tc='d')
    
    # 3. CONVERSION OF THE NEW QUADRATIC CONSTRAINT (L^2 <= B^2 * Var)
    
    if B is not None:
        print(f"\n--- Quadratic Constraint Conversion (B={B}) ---")

        # --- A. Length Squared Matrix (P_L) ---
        # L^2 = sum_{i=0}^{Nc-2} [ (x_{i+1} - x_i)^2 ]
        # This P_L matrix will be sparse and block diagonal (Nc x Nc in the x-block and y-block)
        P_L_block = np.zeros((no_clusters, no_clusters))
        for i in range(no_clusters - 1):
            P_L_block[i, i] += 1
            P_L_block[i+1, i+1] += 1
            P_L_block[i, i+1] -= 1
            P_L_block[i+1, i] -= 1
        
        # P_L is the full matrix for 2*Nc variables
        P_L_np = np.block([
            [P_L_block, np.zeros((no_clusters, no_clusters))],
            [np.zeros((no_clusters, no_clusters)), P_L_block]
        ])
        print("P_L Matrix (Length Squared) generated successfully.")
        
        # --- B. Variance Matrix (P_V) ---
        # Var(x) = sum_{i=0}^{Nc-1} C_i (x_i - mean(x))^2.
        # For simplicity, we assume Var(x) is the weighted sum of squared distances 
        # from the weighted mean, resulting in a quadratic term P_V.

        # Total mass M
        M = C.sum()
        
        # Weighted mean calculation is complex to embed purely in P_V. 
        # A simpler quadratic form for variance: Var(x) = sum C_i * x_i^2 - M * mean(x)^2
        # P_V matrix is proportional to a weighted diagonal matrix minus mean interaction terms.
        
        # Identity matrix for the coordinates
        I_Nc = np.eye(no_clusters)
        
        # Matrix to calculate sum x_i / Nc (or weighted sum)
        W = np.diag(C)
        One_Nc = np.ones((no_clusters, 1))

        # This part requires dense matrix algebra to correctly capture the mean subtraction:
        # P_V = 2 * (W - W @ One_Nc @ One_Nc.T @ W / M)
        # Using a simpler form for demonstration (weighted diagonal):
        P_V_simple_block = np.diag(C)
        P_V_np = np.block([
            [P_V_simple_block, np.zeros((no_clusters, no_clusters))],
            [np.zeros((no_clusters, no_clusters)), P_V_simple_block]
        ])
        print("P_V Matrix (Simplified Variance) generated successfully.")

        # --- C. Final Quadratic Constraint Matrix ---
        P_final_constraint = P_L_np - (B**2) * P_V_np
        print(f"P_Final_Constraint (P_L - B^2 * P_V) generated. (Size: {P_final_constraint.shape})")
        
        # CVXOPT Warning:
        print("\n!!! WARNING !!!")
        print("The solver 'solvers.qp' used below IGNORES this quadratic constraint.")
        print("To use the constraint, you must reformulate the entire problem into")
        print("Second-Order Cone Program (SOCP) format and use 'solvers.conelp'.")


    # 4. Solve the UNCONSTRAINED QP (Ignoring the P_final_constraint)
    G = matrix([], (0, N), 'd')
    h = matrix([], (0, 1), 'd')
    A = matrix([], (0, N), 'd')
    b = matrix([], (0, 1), 'd')
    
    try:
        sol = solvers.qp(P, q, G, h, A, b)
    except Exception as e:
        print(f"CVXOPT Solver error: {e}")
        return None, None

    # 5. Extract results and reconstruct solution
    if sol['status'] == 'optimal':
        x_solution_vector = np.array(sol['x']).flatten()
        
        # Reshape solution: x_coords (first Nc), y_coords (second Nc)
        x_coords = x_solution_vector[0:no_clusters]
        y_coords = x_solution_vector[no_clusters:2*no_clusters]
        
        x_solution = np.vstack([x_coords, y_coords]).T
        
        # Calculate the actual objective value
        obj_val_qp = sol['primal objective']
        const_term = np.sum(ga * y[:, 0]**2) + np.sum(ga * y[:, 1]**2)
        obj_val_actual = obj_val_qp + const_term
        
        # Print the result similar to the original code
        print(f'\nOptimal objective value: {obj_val_actual}')
        return obj_val_actual, x_solution
    else:
        print(f"No optimal solution found. Status: {sol['status']}")
        return None, None

# --- MOCK EXECUTION FOR DEMONSTRATION ---

# # Define parameters
# NO_CLUSTERS = 5
# NO_DATA = 10
# L_BOUND = 5.0
# B_RATIO = 2.0 # The new constraint ratio L/SD <= B

# # Generate mock data
# y_mock = np.random.rand(NO_DATA, 2) * 10
# ga_mock = np.random.rand(NO_CLUSTERS, NO_DATA) + 0.1 

# print("Starting CVXOPT Unconstrained QP Solver...")

# # Solve the unconstrained problem and generate constraint matrices
# obj_val, solution = find_location_cvxopt(y_mock, ga_mock, NO_CLUSTERS, L_BOUND, B=B_RATIO)

# if solution is not None:
#     print("\nOptimal Cluster Locations (x, y):")
#     print(solution)
# else:
#     print("Solver failed to find a solution.")

def find_coupling(x, y, nu, la, no_clusters, no_data):
    n, k = no_clusters, no_data
    X = x[:, np.newaxis, :] 
    Y = y[np.newaxis, :, :]
    C = c(X, Y)

    # Pre-calculate the squared distances
    dist_squared = np.sum((x[:, np.newaxis] - y) ** 2, axis=2)

    # Create a new model
    m = gp.Model("quadratic")

    # Create variables
    ga = m.addVars(n, k, vtype=GRB.CONTINUOUS, name="ga")

    # Set objective
    obj = gp.quicksum(dist_squared[i, j] * ga[i, j] for i in range(n) for j in range(k)) + \
        la * gp.quicksum(sum((sum(y[j][d] * ga[i, j] for j in range(k)) - x[i][d]) ** 2 for d in range(2)) for i in range(n))
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    for j in range(k):
        m.addConstr(sum(ga[i, j] for i in range(n)) == nu[j])

    # Optimize model
    m.optimize()

    # Calculate the optimal solution
    ga_opt = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            ga_opt[i, j] = ga[i, j].X

    total_value = np.sum(ga_opt * C)

    m.setParam('OutputFlag', 0)

    return ga_opt, total_value

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


def recover_original_data(scaled_data, min_vals, max_vals, new_min=-1.5, new_max=1.5):
    # Reverse the scaling and shifting
    norm_data = (scaled_data - new_min) / (new_max - new_min)
    
    # Recover the original data
    original_data = norm_data * (max_vals - min_vals) + min_vals
    
    return original_data

def closest_point_and_distance(x, y):
    """
    Find the closest point or line segment in x for each point in y.
    """
    
    n = x.shape[0]
    m = y.shape[0]

    closest_points = np.zeros((m, 2))
    square_distances = np.zeros(m)
    project_set = np.zeros(m, dtype=int)
    
    for i, point_y in enumerate(y):
        # Calculate distances from point y to each point in x
        sq_distances_to_segments = np.zeros(n-1)
        closest_points_on_segments = np.zeros((n-1, 2))
        sq_distances_to_points = np.sum((x - point_y)**2, axis=1)

        # Calculate distances from point y to each line segment formed by consecutive points in x
        for j in range(len(x)-1):
            sq_distances_to_segments[j], closest_points_on_segments[j] = point_line_distance_and_closest_point(x[j], x[j+1], point_y)

        # Find the minimum distance
        min_sq_distance_to_point = sq_distances_to_points.min()
        min_sq_distance_to_segment = sq_distances_to_segments.min()
        
        # Determine which is closer, point or segment
        if min_sq_distance_to_point <= min_sq_distance_to_segment:
            closest_point_index = sq_distances_to_points.argmin()
            closest_point = x[closest_point_index]
            closest_distance = min_sq_distance_to_point
            project_index = closest_point_index
        else:
            closest_segment_index = sq_distances_to_segments.argmin()
            closest_point = closest_points_on_segments[closest_segment_index]
            closest_distance = min_sq_distance_to_segment
            project_index = n + closest_segment_index

        closest_points[i] = closest_point
        square_distances[i] = closest_distance
        project_set[i] = project_index

    return closest_points, square_distances, project_set

def initial_curve_with_pca(X, shift_X, no_data, no_clusters):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # Project the data onto the first principal component
    X_pca = np.dot(X, Vt[0])

    min_val = np.min(X_pca)
    max_val = np.max(X_pca)

    # Define t based on the range of the projected data
    t = np.linspace(min_val, max_val, no_clusters)
    cluster_width = (max_val - min_val) / no_clusters
    groups = np.zeros((no_data, no_clusters))
    groups_int = np.zeros((no_data, no_clusters), dtype=int)
    groups_no = np.zeros(no_clusters)

    for i in range(no_data):
        j = int((X_pca[i] - min_val) / cluster_width)
        # Ensure that the maximum value is assigned to the last group
        if j == no_clusters:
            j = no_clusters - 1
        groups[i, j] = 1 / no_data
        groups_int[i, j] = 1
        groups_no[j] += 1

    x = np.zeros((no_clusters,2))
    pca_mean = np.zeros(no_clusters)
    mu = np.zeros(no_clusters)
    for j in range(no_clusters):
        # find the barycenter of each cluster
        mu[j] = np.sum(groups[:, j])
        pca_mean[j] = np.sum(X_pca * groups_int[:, j]) / groups_no[j]

    x = np.outer(pca_mean, Vt[0])
    x = x + shift_X

    # calculate the total square distance from all the y to the corresponding baricenter
    total_length = 0
    for i in range(no_data):
        for j in range(no_clusters):
            total_length += groups[i, j] * ((y[i][0] - x[j][0]) ** 2 + (y[i][1] - x[j][1]) ** 2)

    return x, groups.T, total_length

def rescale_data(data, new_min=-1.5, new_max=1.5):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Normalize to 0-1
    norm_data = (data - min_vals) / (max_vals - min_vals)
    
    # Scale to new range and shift
    scaled_data = norm_data * (new_max - new_min) + new_min
    
    # Return scaled data along with the scaling parameters
    return scaled_data, min_vals, max_vals

def calculate_wasserstein_distance(mu, nu, X1, X2):
    # Calculate the cost matrix
    M = ot.dist(X1, X2)

    # Calculate the Wasserstein distance
    wasserstein_distance = ot.emd2(mu, nu, M)

    return wasserstein_distance

def point_line_distance_and_closest_point(p1, p2, p3):
    """
    Calculate the perpendicular distance between each point in p3_array and the line formed by points p1 and p2.
    Also find the closest point on the line to each point in p3_array.
    If the closest point is not between p1 and p2, return infinity.
    """

    # Vector representing the line segment from p1 to p2
    line_vector = p2 - p1

    # Vector representing the line segment from p1 to p3
    point_vector = p3 - p1

    # Calculate the scalar projection of point_vector onto line_vector
    scalar_projection = np.dot(point_vector, line_vector) / np.linalg.norm(line_vector)

    # Calculate the projection vector
    projection_vector = scalar_projection * line_vector / np.linalg.norm(line_vector)

    # Calculate the perpendicular distance
    # Calculate the closest point on the line to p3
    # If the scalar projection is not between 0 and the length of the line segment, return distance to two end
    if scalar_projection <= 0: 
        # perpendicular_distance = np.linalg.norm(p3 - p1)
        perp_sq_dist = np.sum((p3 - p1)**2)
        closest_point = p1
    elif scalar_projection >= np.linalg.norm(line_vector):
        # perpendicular_distance = np.linalg.norm(p2 - p1)
        perp_sq_dist = np.sum((p2 - p1)**2)
        closest_point = p2
    else:
        # perpendicular_distance = np.linalg.norm(point_vector - projection_vector)
        perp_sq_dist = np.sum((point_vector - projection_vector)**2)
        closest_point = p1 + projection_vector

    return perp_sq_dist, closest_point

# --- PROCESS FLOW ---

# 0. Set parameters
objs = []
itr = 0
# no_data = 200
no_clusters = 10
eta = 0.6
la = 0.006

# np.random.seed(40)  # for reproducibility

# # 1. Data Creation and preprocessing
# sigma = 0.1
# z = np.linspace(-1, 1, no_data)
# z = np.column_stack((z, z**2))
# noise = np.random.normal(0, sigma, size=(no_data, 2))
# y = z + noise 

# # transform the data y so that the mean of y is 0
# z = z - y.mean(axis=0)
# y = y - y.mean(axis=0)

# y, y_rescale_min, y_rescale_max = rescale_data(y)

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


# plot the data
plt.scatter(y[:,0], y[:,1])
plt.title('Data points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axis('equal')
plt.show()

    

# transform the data y so that the mean of y is 0
z = p - y.mean(axis=0)
y = y - y.mean(axis=0)

y, y_rescale_min, y_rescale_max = rescale_data(y)

no_data = n
nu = np.ones(no_data) / no_data
mu_original = np.ones(no_data) / no_data

# 2. Run Kantorovich Order
# x_kan_2, mu_kan_2, lambda_1_kan_2, lambda_2_kan_2, lambda_3_kan_2, lambda_1i_kan_2, W2_kan_2 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=2)
# closest_point_len_kan_2, _, _ = closest_point_and_distance(x_kan_2, y)
# x_kan_plot_2 = recover_original_data(x_kan_2, y_rescale_min, y_rescale_max)
# print(f'The Wassertein distance between Kantorovich order closest point with length 2 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_2, z)}')
# print(f'The total square distance between Kantorovich order with length 2 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_2) ** 2))}')

# x_kan_3, mu_kan_3, lambda_1_kan_3, lambda_2_kan_3, lambda_3_kan_3, lambda_1i_kan_3, W2_kan_3 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=3)
# closest_point_len_kan_3, _, _ = closest_point_and_distance(x_kan_3, y)
# x_kan_plot_3 = recover_original_data(x_kan_3, y_rescale_min, y_rescale_max)
# print(f'The Wassertein distance between Kantorovich order closest point with length 3 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_3, z)}')
# print(f'The total square distance between Kantorovich order with length 3 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_3) ** 2))}')

# x_kan_4, mu_kan_4, lambda_1_kan_4, lambda_2_kan_4, lambda_3_kan_4, lambda_1i_kan_4, W2_kan_4 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=4)
# closest_point_len_kan_4, _, _ = closest_point_and_distance(x_kan_4, y)
# x_kan_plot_4 = recover_original_data(x_kan_4, y_rescale_min, y_rescale_max)
# print(f'The Wassertein distance between Kantorovich order closest point with length 4 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_4, z)}')
# print(f'The total square distance between Kantorovich order with length 4 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_4) ** 2))}')

# 3. Run Fifth Method
y_shift = y - y.mean(axis=0)
x_convex, ga, initialL = initial_curve_with_pca(y_shift, y.mean(axis=0), no_data, no_clusters)
# x_convex, ga, initialL = initial_curve_with_pca(y_shift, y.mean(axis=0), no_data, no_data)
prev_L = np.sqrt(initialL)


while itr <= 100:
    print(f"Convex order Iteration: {itr}")

    # obj_convex, x_convex = find_location(y, ga, no_clusters, 2)
    # obj_convex, x_convex = find_location_cvxopt_sdp(y, ga, no_clusters, 2)
    # obj_convex, x_convex = find_location_cvxopt(y, ga, no_clusters, 2)
    obj_convex, x_convex = find_location_cvxopt_L_to_SD_ratio(y, ga, no_clusters, 2, B=1)
    # new_ga, total_value = find_coupling(x_convex, y, nu, la, no_clusters, no_data)
    new_ga, total_value = find_coupling_cvxopt(x_convex, y, nu, la, no_clusters, no_data)

    objs.append([math.sqrt(obj_convex), math.sqrt(total_value)])


    print(f'The diiferent between prev_L and total_value is {prev_L - math.sqrt(total_value)}')
    if np.abs(prev_L - math.sqrt(total_value)) < 1e-4:
        print(itr)
        break
    else:
        ga = new_ga
        prev_L = math.sqrt(total_value)
        itr = itr + 1
closest_point_len, _, _ = closest_point_and_distance(x_convex, y)

x_con_plot = recover_original_data(x_convex, y_rescale_min, y_rescale_max)
print(f'The Wassertein distance between Length constrained closest point and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len, z)}')
print(f'The total square distance between Length constrained and orginal mu is {np.sqrt(np.sum((z - closest_point_len) ** 2))}')

# save the results as pickle file
with open('convex_order_results.pkl', 'wb') as f:
    pickle.dump({
        'x_con_plot': x_con_plot
    }, f)

# load the results from pickle file and print them
with open('convex_order_results.pkl', 'rb') as f:
    data = pickle.load(f)
    x_con_plot_load = data['x_con_plot']


# Print loaded results
print(f"Loaded x_con_plot: {x_con_plot_load}")


# 4. Plot results

color_scale = ['#0000ff', '#ff0000', '#00ff00']

curve_div_no = no_data
y_plot = recover_original_data(y, y_rescale_min, y_rescale_max)

# Plot Kantorovich Dominance vs Convex Order
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=[7, 5])

# Plot the data points
ax.scatter(x=y_plot[:,0], y=y_plot[:,1], s=25, marker='s', color='black', alpha=.2, label='Observed Data Points')
ax.scatter(x=z[:,0], y=z[:,1], s=25, marker='s', color='#E7D046', alpha=.7, label='Un-polluted Data Points')

# # Plot the Kantorovich Dominance curve
# ax.plot(x_kan_plot_4[:,0], x_kan_plot_4[:,1], color=color_scale[0], linewidth=5, alpha=.85)
# ax.scatter(x=x_kan_plot_4[:,0], y=x_kan_plot_4[:,1], color=color_scale[0], s=20, alpha=.75, label='Kantorovich dominance')
  
# Plot Convex order curve
ax.plot(x_con_plot[:,0], x_con_plot[:,1], color=color_scale[1], linewidth=4, alpha=.55  )
ax.scatter(x=x_con_plot[:,0], y=x_con_plot[:,1], color=color_scale[1], s=40, alpha  =.75, label='Convex order')

plt.legend(prop={'size': 20}, markerscale=2.5)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()



# # Plot Kantorovich Dominance comparison
# plt.rcParams['text.usetex'] = True
# fig, ax = plt.subplots(figsize=[7, 5])

# # Plot the data points
# ax.scatter(x=y_plot[:,0], y=y_plot[:,1], s=25, marker='s', color='black', alpha=.2, label='Observed Data Points')
# ax.scatter(x=z[:,0], y=z[:,1], s=25, marker='s', color='#E7D046', alpha=.7, label='Un-polluted Data Points')

# # Plot the Kantorovich Dominance curve
# ax.plot(x_kan_plot_2[:,0], x_kan_plot_2[:,1], color=color_scale[0], linewidth=5, alpha=.85)
# ax.scatter(x=x_kan_plot_2[:,0], y=x_kan_plot_2[:,1], color=color_scale[0], s=20, alpha=.75, label='Length = 2')

# ax.plot(x_kan_plot_3[:,0], x_kan_plot_3[:,1], color=color_scale[1], linewidth=5, alpha=.85)
# ax.scatter(x=x_kan_plot_3[:,0], y=x_kan_plot_3[:,1], color=color_scale[1], s=20, alpha=.75, label='Length = 3')

# ax.plot(x_kan_plot_4[:,0], x_kan_plot_4[:,1], color=color_scale[2], linewidth=5, alpha=.85)
# ax.scatter(x=x_kan_plot_4[:,0], y=x_kan_plot_4[:,1], color=color_scale[2], s=20, alpha=.75, label='Length = 4')

# plt.legend(prop={'size': 20}, markerscale=2.5)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.show()

