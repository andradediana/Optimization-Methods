# Frank-Wolfe Methods for Recommender Systems

Low-rank matrix completion with the **Frank-Wolfe**, **Pairwise Frank-Wolfe** and **Projected Gradient** algorithms, tested on movie, joke and product rating matrices.

**Authors:** Diego A. Brule Galleguillos, Diana C. Andrade Damian, Evgeni Markin, Marina Lima Braga

University of Padova, final project for ODS, July 2025

---

## Overview

Recommender systems (Netflix, Spotify, TikTok and so on) can be framed as a **sparse matrix completion problem**: given a user × item matrix with many missing ratings, fill in the missing entries while keeping the matrix low-rank, so that users with similar tastes get similar recommendations.

The rank constraint is non-convex, so the project solves the convex relaxation with a **nuclear-norm constraint**:

```
min_X   sum_{(i,j) in Ω} (X_ij - R_ij)^2      s.t.   ||X||_* <= τ
```

Frank-Wolfe methods are a good fit for this problem. They are projection-free: each step needs only the top singular vector pair of the gradient (a cheap linear minimisation oracle), and every iterate stays inside the nuclear-norm ball, so no expensive SVD-based projection is needed. The project compares them with a projected gradient method that does need that projection.

### Research questions

- How do the projection-free FW and PFW methods compare with Projected Gradient Descent (PGD) in accuracy and convergence?
- How does performance change as the data gets sparser?

The full write-up is in [`Group_1_-_Recommender_system.pdf`](Group_1_-_Recommender_system.pdf).

---

## Repository contents

| File | Description |
|---|---|
| `recommender_systems_group_1.py` | All three algorithms, helper functions, and the experiments on the three datasets (exported from a Colab notebook) |
| `Group_1_-_Recommender_system.pdf` | Project report |
| `lens_movies_short.csv` | MovieLens-style ratings: top 300 most active users × 30 most rated movies (no header) |
| `jester_dataset.csv` | Jester joke ratings: 50,692 users × 150 jokes (no header, ~36 MB) |
| `ratings_matrix.csv` | Amazon product ratings: 622 users × 827 products (header row and a `user_id` column) |

## Algorithms

| Algorithm | Idea |
|---|---|
| **Frank-Wolfe (FW)** | Take the rank-1 atom `τ·u₁v₁ᵀ` from the top singular vectors of `−gradient` and move toward it with an exact line search. |
| **Pairwise Frank-Wolfe (PFW)** | Keeps an active set of atoms and moves weight from the worst "away" atom to the best new FW atom, with a step size bounded by the away atom's weight. |
| **Projected Gradient Descent (PGD)** | Takes a gradient step, then projects back onto the nuclear-norm ball: SVD, then l1-ball projection of the singular values. |

Implementation notes:
- The gradient is computed with a **mask**, so only originally observed entries contribute to the loss.
- The linear minimisation oracle uses `scipy.sparse.linalg.svds` (top singular pair only).
- FW and PFW use **line search** instead of the fixed `2/(k+2)` step size, which gave better results.
- Active-set atoms in PFW are de-duplicated with a tolerance.

## Experimental setup

1. **Normalise** ratings (divide by 5 for Lens and Amazon, by 10 for Jester).
2. **Split** the observed entries 90% train / 10% test (`random_state=42`).
3. **Tune** hyperparameters on the training data.
4. **Evaluate** on held-out entries after scaling back to the original rating scale.

Metrics: MAE on all datasets, plus exact and rounded accuracy where ratings are discrete.

### Datasets

| Dataset | Size | Missing entries | Rating range |
|---|---|---|---|
| Lens Movies (short) | 300 × 30 | ~43% | 0.5 – 5 |
| Jester | 50,692 × 150 | ~77% | −10 – 10 (continuous) |
| Amazon Products | 622 × 827 | ~99.4% | 1 – 5 |

### Hyperparameters used in the script

| Dataset | FW / PFW (τ, iterations) | PGD (τ, learning rate, iterations) |
|---|---|---|
| Lens Movies | 100, 100 | 100, 0.1, 100 |
| Jester | 250, 100 | 50, 0.1, 100 |
| Amazon | 1000, 300 | 100, 1, 300 |

Predictions are clipped (and, for discrete ratings, rounded) before computing accuracy: to the nearest 0.5 in [0.5, 5] for Lens, and to integers in [1, 5] for Amazon. "Round accuracy" on Lens compares ratings rounded to whole points.

---

## Results

### Lens Movies

| Algorithm | MAE | Exact accuracy | Round accuracy |
|---|---|---|---|
| Frank-Wolfe | 0.58 | 0.27 | 0.52 |
| Pairwise Frank-Wolfe | 0.60 | 0.26 | 0.51 |
| Projected Gradient | 0.59 | 0.25 | 0.52 |

### Jester

| Algorithm | MAE |
|---|---|
| Frank-Wolfe | 0.87 |
| Pairwise Frank-Wolfe | 0.87 |
| Projected Gradient | 0.36 |

### Amazon Products

| Algorithm | MAE | Exact accuracy |
|---|---|---|
| Frank-Wolfe | 1.79 | 0.44 |
| Pairwise Frank-Wolfe | 1.62 | 0.48 |
| Projected Gradient | 1.95 | 0.24 |

### Main findings

- **Convergence:** FW and PFW converge very quickly. PGD needs more iterations on the Lens data, as expected from its extra projection step.
- **Lens Movies (least sparse):** all three methods give similar MAE and accuracies.
- **Jester:** FW and PFW reach lower training loss than PGD, but PGD shows a lower reported MAE.
- **Amazon (most sparse, 0.6% observed):** results are the weakest, as expected, with PFW best on MAE and exact accuracy and PGD worst.
- **Overall:** the projection step can slow convergence and does not guarantee better results. FW and PFW are good projection-free alternatives with metrics comparable to PGD, sometimes better, and fast convergence.

---

## How to run

The script was exported from a Google Colab notebook, so it runs as a plain Python script or can be pasted into a notebook.

1. Clone the repository and keep the `.py` file and the three `.csv` files in the **same folder**. The script reads them with relative paths.
2. Install the requirements:
   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib tqdm
   ```
3. Run:
   ```bash
   python recommender_systems_group_1.py
   ```
   The script processes the datasets in order (Lens → Jester → Amazon), prints the metrics for each algorithm and plots the loss convergence curves.

**Runtime:** Lens and Amazon are fast. Jester (50,692 × 150) is much heavier because it runs repeated SVD-based steps on a dense matrix. Expect it to take noticeably longer and use more memory.

## Known issues and notes

- Results come from a single random split (`random_state=42`) and no repeated runs or cross-validation.
- Hyperparameters (τ, learning rate, iterations) were tuned by hand and differ per dataset and method.

## References

- Frank, M., & Wolfe, P. (1956). An algorithm for quadratic programming. *Naval Research Logistics Quarterly*.
- Jaggi, M. (2013). Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization. *ICML*.
- Lacoste-Julien, S., & Jaggi, M. (2015). On the Global Linear Convergence of Frank-Wolfe Optimization Variants. *NeurIPS*.
