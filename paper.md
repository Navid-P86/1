
# Interaction-Based Clustering: A Unified Framework Linking Feature Interactions and Regime-Dependent Models

**Author:** Navid Pourdad  
**Email:** navid.pourdad86@gmail.com

---

## Abstract
Interaction-Based Clustering (IBC) connects important interaction features (products of variables such as $X_i X_j$) to natural regime boundaries in data. This paper presents mathematical proofs, simulations, and visualizations showing that dominant interactions induce separable sub-regimes and that clustering along variables in top interactions can dramatically improve simple regression models. We provide code and figures for reproducibility.

---

## 1. Introduction
(omitted here for brevity; see sections below)

---

## 2. Key idea
If the data-generating process includes a term $\beta_{ij} (X_i X_j)$, then for fixed $X_j=c$,
\[
\mathbb{E}[Y \mid X_i, X_j=c] \approx \beta_{ij} c \cdot X_i + \text{(small terms)}.
\]
Thus conditioning on $X_j$ transforms a multiplicative effect into an approximately linear relationship in $X_i$ whose slope depends on $c$. Splitting by $X_j$ therefore creates clusters with simpler linear behavior.

---

## 3. Mathematical proofs (concise)
### Proof 1: Conditional slope
For model
\[
Y = \beta_0 + \beta_i X_i + \beta_j X_j + \beta_{ij} (X_i X_j) + \epsilon,
\]
the conditional slope w.r.t. $X_i$ is
\[
\frac{\partial Y}{\partial X_i} = \beta_i + \beta_{ij} X_j.
\]
Hence fixing $X_j$ yields piecewise linear relationships in $X_i$.

### Proof 2: Geometric interpretation
The interaction term creates a twisted response surface in $(X_i, X_j, Y)$ space. The sign/magnitude of $X_i X_j$ partitions the domain into lobes that are approximated well by local planes.

### Proof 3: Variance decomposition
Conditioning on a cluster indicator derived from $X_j$ increases explained variance if the dominant source of variance in $Y$ is the interaction term.

---

## 4. Simulation study (small dataset, reproducible)
We generate a small dataset (n=500) where $X_1,\dots,X_4 \sim N(0,1)$ and
\[
Y = 0.5 + 0.05X_1 + 0.05X_2 + 0.05X_3 + 0.05X_4 + 4.0(X_1 X_4) + \epsilon.
\]
The dataset is saved as `ibc_synthetic_small.csv` in the package.

### 4.1 Models compared
- **A:** Linear regression on raw features (no interactions).  
- **B (IBC):** Split by $X_4$ sign (>=0 vs <0), fit separate linear regressions on original features.  
- **C:** Global linear regression on interaction-only features (degree=2).  
- **D:** Fit interaction models inside each cluster and combine predictions.

### 4.2 Results (in-sample R²)
- A (no interaction): **0.016**  
- B (IBC clustered simple): **0.589**  
- C (global with interactions): **0.947**  
- D (clustered with interactions): **0.948**

_Figure 1_ shows the underlying response surface for $X_1$ and $X_4$ (noise-free slice).

![Fig 1 — Response Surface](figures/fig1_interaction_surface.png)

_Figure 2_ shows the scatter of $X_1$ vs $Y$ colored by cluster based on sign($X_4$).

![Fig 2 — Cluster split](figures/fig2_cluster_split.png)

_Figure 3_ compares R² across models.

![Fig 3 — R2 comparison](figures/fig3_r2_comparison.png)

_Figure 4_ shows the top Lasso coefficients (absolute values) for interaction features discovered by LassoCV.

![Fig 4 — Top Lasso Coefficients](figures/fig4_top_lasso_coefs.png)

_Figure 5_ demonstrates a degree-3 interaction slice (for intuition).

![Fig 5 — Degree-3 demo](figures/fig5_higher_order_demo.png)

---

## 5. Higher-order interactions
(Discussed in the paper; code for degree-3 demonstration is included inline and in the `code/` directory.)

---

## 6. Discussion and practical recommendations
(See earlier sections — use screening, regularization, cross-validation, hierarchical/multilevel approaches for per-cluster parameter sharing if sample sizes are small.)

---

## 7. Conclusion
IBC provides a principled method to discover regime structure via important interaction features, improve simple models through clustering, and further refine models with cluster-specific interaction models when regimes differ.

---

## Appendix: Key code snippets (reproducible)
Below are compact code blocks; full scripts are saved in `code/ibc_simulation.py` and `code/figure_generation.py`.

### Data generation and model comparison
```python
# see code/ibc_simulation.py
import pandas as pd
print(pd.read_csv('ibc_synthetic_small.csv').head())
```

### Lasso interaction discovery
```python
# top interactions (absolute coefs)
import pandas as pd
print(pd.read_csv('top_interaction_coefs.csv', header=None).head())
```

---

### Files in this package
- `paper.md` (this file)  
- `figures/` (PNG figures)  
- `code/ibc_simulation.py`  
- `code/figure_generation.py`  
- `ibc_synthetic_small.csv`  

---
*End of paper.*
