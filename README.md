# Data Denoising via Optimal Transport

Examples of our approach to Principal Curves based on Optimal Transport.

Problem: Given a set of points $y=(y_j)$ with probability distribution $v=(v_j)$, we compute a representative set $x=(x_i)$, its probability $u=(u_i)$ and an optimal transport map $\pi=(\pi_{ij})$, $j=1,\ldots,n$, $i=1,\ldots,m$, $m\leq n$. $x$ satisfies constraints pertaining to the problem domain. $\pi$ minimizes the Wasserstein distance between $x$ and $y$, and the error term $\varepsilon=(\varepsilon_{ij})$ satisfies the orthogonality condition $\sum_i \sum_j \pi_{ij} x_j e_{ij} = 0$.

Example 1 -- Bounded curvature. The amount of curvature $\phi(x)$ is bounded. The resulting set forms a curve passing through the data. If the curvature tolerance goes to zero, the solution approximates the projected points of the PCA problem. If the curvature tolerance is big enough that it becomes non-binding, the problem becomes equivalent to the $K$-means.

Foundational paper and theoretical support to be made public soon. Other examples may be added.
