# Data Denoising via Optimal Transport

Examples of our approach to Principal Curves based on Optimal Transport.

Problem: Given a set of points $y=(y_j)$ with probability distribution $v=(v_j)$, we compute a representative set $x=(x_i)$, its probability $u=(u_i)$ and an optimal transport map $\pi=(\pi_{ij})$, $j=1,\ldots,n$, $i=1,\ldots,m$, $m\leq n$. $x$ satisfies constraints pertaining to the problem domain. $\pi$ minimizes the Wasserstein distance between $x$ and $y$, and the error term $\varepsilon=(\varepsilon_{ij})$ satisfies the orthogonality condition $\sum{i} \sum{j} pi_{ij} x_j e_{ij} = 0$.

Example 1 -- Bounded curvature. The amount of curvature of $x$, calculated by an appropriate function $\phi(x)$, is bounded. The resulting set forms a curve passing through the data and whose curvature is controlled by a parameter. If the curvature tolerance approximates zero, the solution approximates the projected points of the PCA problem. If the curvature tolerance is big enough that it becomes non-binding, the problem becomes equivalent to the $K$-means. Details, theoretical support and additional examples will be published by June, 2024.

Foundational paper and theoretical support to be made public soon. Other examples may be added.
