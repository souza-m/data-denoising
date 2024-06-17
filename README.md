# Data Denoising via Optimal Transport

Problem: Given a set of points $y=(y_j)\in\mathbb{R}^d$ with probability distribution $v=(v_j)$, we compute a representative set $x=(x_i)$, its probability $u=(u_i)$ and an optimal transport map $\pi=(\pi_{ij})$, $j=1,\ldots,n$, $i=1,\ldots,m$, $m\leq n$. $x$ satisfies constraints pertaining to the problem domain (see examples); $\pi$ minimizes the Wasserstein distance between $x$ and $y$; and the error term $\varepsilon=(\varepsilon_{ij})$, $\varepsilon_{ij}=y_j-x_i$, has mean zero and satisfies the orthogonality condition $\sum_i \sum_j \pi_{ij}  \langle x_j,e_{ij}\rangle = 0$.

Example 1 -- Bounded curvature. The amount of curvature measured by an appropriate function $\phi(x)$ is penalized. The resulting set forms a Principal Curve passing through the data that can be more or less curvy depending on a penalization parameter. If the penalization is too large, the curve becomes straight and the solution approximates the projected points of the PCA problem. If the penalization parameter is small enough that it becomes non-binding, the problem becomes equivalent to the $K$-means.

Foundational paper and theoretical support to be made public soon. Other examples may be added.
