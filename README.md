# Data Denoising via Optimal Transport

Given a set of points $y=(y_j)\subset\mathbb{R}^d$ with probability distribution $v=(v_j)$, we compute a representative set $x=(x_i)\subset\mathbb{R}^d$, its distribution $u=(u_i)$ and an optimal transport map (joint distribution) $\pi=(\pi_{ij})$, $j=1,\ldots,n$, $i=1,\ldots,m$, $m\leq n$. $x$ satisfies constraints pertaining to the problem domain (see examples); $\pi$ minimizes the Wasserstein distance between $x$ and $y$; and the error (or noise) term $\varepsilon=(\varepsilon_{ij})$, $\varepsilon_{ij}=y_j-x_i$, has mean zero and satisfies the orthogonality condition $\sum_i \sum_j \pi_{ij}  \langle x_j,e_{ij}\rangle = 0$.

Example 1 - Bounded curvature. The amount of curvature measured by an appropriate function $\phi(x)$ is bounded/penalized. The resulting set forms a Principal Curve passing through the data that can be more or less curvy depending on a penalization parameter. If $m=n$ and the penalization is large, the curve becomes straight and the solution approximates the projected points of the PCA problem. Case $m < n$ to be added.

Example 2 - $K$-means with fixed weights. Equivalent to the $K$-means with $K=m$, but all centroids have the given fixed weight (in this example, $1/m$).

Foundational paper and theoretical support to be made public soon. Other examples may be added.
