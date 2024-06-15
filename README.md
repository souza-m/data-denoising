# Data Denoising via Optimal Transport

This is the numerical implementation of a principal cuve with orthogonal noise and bounded curvature. If the curvatuere bound goes to zero, or equivalently, the penalization term goes grows arbitrarily, the solution exactly approximates the projection points in the PCA method.

Main problem: given a set of points $y=y_1,\ldots,y_n$, we compute a representative set $x=x_1,\ldots,x_m$, %m<n$, such that $x$ generates $y$ in some sense -- an optimal transport map associates the $x$-points to the $y$-points, and the residuals $y-x$ are orthogonal to $x$. Visually, the resulting set is a curve that passes through the data and has limited curvature. If the curvature tolerance approximates zero, or equivalently, the penalty term goes to infinity, the solution approximates the projected points of the PCA problem.

Details, theoretical support and additional examples will be published by June, 2024.
