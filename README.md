# Data Denoising via Optimal Transport

This repository contains the implementation of the numerical section associated with the research paper **"Data Denoising via Optimal Transport and Kantorovich Dominance"** (to be published). This project introduces a novel framework for data denoising inspired by martingale optimal transport, with applications in robust data analysis, dimensionality reduction, clustering, and more.

---

## Abstract
We introduce a new framework for data denoising, partially inspired by martingale optimal transport. For a given noisy distribution (the data), our approach involves finding the closest distribution to it among all distributions which:
1. Have a particular prescribed structure (expressed by requiring they lie in a particular domain).
2. Are self-consistent with the data.

We show that this amounts to maximizing the variance among measures in the domain which are dominated in convex order by the data. For particular choices of the domain, this problem and a relaxed version of it, in which the self-consistency condition is removed, are intimately related to various classical approaches to denoising. We prove that our general problem has certain desirable features: solutions exist under mild assumptions, have certain robustness properties, and, for very simple domains, coincide with solutions to the relaxed problem.

We also introduce a novel relationship between distributions, termed **Kantorovich dominance**, which retains certain aspects of the convex order while being a weaker, more robust, and easier-to-verify condition. Building on this, we propose and analyze a new denoising problem by substituting the convex order in the previously described framework with Kantorovich dominance. We demonstrate that this revised problem shares some characteristics with the full convex order problem but offers enhanced stability, greater computational efficiency, and, in specific domains, more meaningful solutions. Finally, we present simple numerical examples illustrating solutions for both the full convex order problem and the Kantorovich dominance problem.

---

## Features
This project implements the numerical approaches discussed in the paper, including:
- **Optimal Transport Framework**: Solves data denoising problems using optimal transport principles.
- **Convex Order and Kantorovich Dominance**: Implements models based on structural constraints and dominance relationships for robust denoising.
- **Proof-of-Concept Examples**: Demonstrates the framework using simple numerical examples, including:
  - **Bounded Curvature**: Principal curves with curvature constraints.
  - **Weighted K-means**: Clustering with fixed centroid weights.

---

## Mathematical Framework
Given a set of points \( y = (y_j) \in (\mathbb{R}^d)^n \) with probability distribution \( v = (v_j) \), this framework computes:
1. A representative set \( x = (x_i) \in (\mathbb{R}^d)^m \),
2. Its distribution \( u = (u_i) \), and
3. An optimal transport map (joint distribution) \( \pi = (\pi_{ij}) \), where \( j = 1, \ldots, n \), \( i = 1, \ldots, m \), with \( m \leq n \).

The solution satisfies:
- Constraints pertaining to the problem domain (e.g., bounded curvature, clustering).
- \( \pi \) minimizes the Wasserstein distance between \( x \) and \( y \).
- The error (or noise) term \( \varepsilon = (\varepsilon_{ij}) \), \( \varepsilon_{ij} = y_j - x_i \), has mean zero and satisfies the orthogonality condition:
  \[
  \sum_i \sum_j \pi_{ij} \langle x_j, e_{ij} \rangle = 0.
  \]

---

## Examples
### Example 1: Bounded Curvature
The amount of curvature measured by an appropriate function \( \phi(x) \) is bounded or penalized. The resulting set forms a **Principal Curve** passing through the data that can be more or less curvy depending on a penalization parameter. 
- If \( m = n \) and the penalization is large, the curve becomes straight and the solution approximates the projected points of the PCA problem.
- Spline functionality will be added for graphic applications.

### Example 2: Weighted K-means
This example is equivalent to the **K-means algorithm** with \( K = m \), but centroids have fixed weights (e.g., \( 1/m \)).
- The order of the \( x \)-points is not relevant.
- If an order is attributed (e.g., to form a path of minimum distance as in the **Travelling Salesman Problem**), this method can be used to draw a **Principal Curve**.

---

## Installation
To set up the project environment, use the provided `requirements.txt` file to install necessary dependencies:
```bash
pip install -r requirements.txt
```

Make sure you have Python 3.8 or later installed.

---

## Project Structure
```plaintext
|   .gitignore
|   LICENSE
|   README.md
|   requirements.txt
|
+---data
|       airquality.csv
|
+---graphics
|       airquality_curves_compare.png
|       airquality_curves_compare_horiz.png
|
+---lib
|       sinkhorn.py         # Third-party implementation for Sinkhorn distances
|
+---src
|       wkm.py              # Core implementation of the framework
|
+---tests
|       curve_examples.py   # Proof-of-concept examples and testing
```

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/souza-m/data-denoising.git
   cd data-denoising
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the examples:
   ```bash
   python tests/curve_examples.py
   ```

---

## References
1. Research Paper: **"Data Denoising via Optimal Transport and Kantorovich Dominance"** (to be published).
2. Martingale Optimal Transport: A foundational concept for this framework.

---

## License
This project is licensed under the [MIT License](./LICENSE).

---

## Contact
For questions or issues, please email `marcelo.souza@bcb.gov.br`.
