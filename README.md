# Data Denoising via Optimal Transport


Example of principal cuve with orthogonal noise and bounded curvature. If the curvatuere bound goes to zero, or equivalently, the penalization term goes grows arbitrarily large, the solution approximates the projection points of the PCA method.

Other examples of data denoising may be added as the paper progresses.

Problem: Given a set of points $y=y_1,\ldots,y_n$, we compute a representative set $x=x_1,\ldots,x_m$, %m<n$ and an optimal "transport map", or joint probability that associates the $y$-points to the $x$-points in such a way that minimizes the Wasserstein distance between $x$ and $y$. The residuals $y-x$ are orthogonal to $x$ under the transport map. Visually, the resulting set is a curve that passes through the data and has limited curvature. If the curvature tolerance approximates zero, or equivalently, the penalty term goes to infinity, the solution approximates the projected points of the PCA problem. If the curvature tolerance is big enough, that is, does not affect the optimization, the problem becomes equivalent to the $K$-means. Details, theoretical support and additional examples will be published by June, 2024.
