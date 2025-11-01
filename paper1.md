Interaction-Based Clustering (IBC) — A Step-by-Step Article
Abstract

Interaction-Based Clustering (IBC) is a practical and theoretical framework linking important interaction features (products of variables like 
𝑋
𝑖
×
𝑋
𝑗
X
i
	​

×X
j
	​

) to natural clusters in data. When an interaction dominates the data-generating process, one variable of that interaction (or a combination of them) can define regime boundaries. Splitting data along those regimes and fitting local models often yields large improvements in predictive performance and interpretability. This article builds the idea from first principles, provides three mathematical proofs, demonstrates them with reproducible code, extends the concept to higher-degree interactions, and outlines how to apply IBC to high-dimensional real-world data.

Table of contents

Notation & setup

The central idea (informal)

Proof 1 — Piecewise linear / conditional-slope argument

Proof 2 — Geometric (surface) interpretation

Proof 3 — Variance decomposition & residual conditioning

Expected 
𝑅
2
R
2
 improvement (analytic intuition + formula)

From theory to practice — algorithmic IBC pipeline

Simulations and experiments (reproducible code)

A/B/C comparison (no-interaction, IBC, interaction model)

D (cluster-specific interaction models)

Extension: degree-3 example

When D₁/D₂ can beat the global interaction model

High-dimensional & real-world considerations (finance, genomics, social)

Practical recommendations & diagnostics

Conclusion and future work

Appendix — code snippets

1. Notation & setup

Features: 
𝑋
1
,
𝑋
2
,
…
,
𝑋
𝑝
X
1
	​

,X
2
	​

,…,X
p
	​

.

Target: 
𝑌
Y.

Interaction of order 2: 
𝑋
𝑖
×
𝑋
𝑗
X
i
	​

×X
j
	​

. Higher-order interaction of order 
𝑑
d: 
∏
𝑚
=
1
𝑑
𝑋
𝑘
𝑚
∏
m=1
d
	​

X
k
m
	​

	​

.

We denote a dominant interaction as one with large absolute coefficient in the true generating process or in a selected model (e.g., large Lasso coefficient).

Models compared:

A — Linear model on original features (no interactions).

B (IBC) — Split dataset by a rule derived from an interaction variable (e.g., sign or quantile of 
𝑋
𝑗
X
j
	​

), fit separate linear regressions per cluster, combine predictions.

C — Global linear model including the interaction feature(s) (e.g., include 
𝑋
𝑖
𝑋
𝑗
X
i
	​

X
j
	​

).

D₁/D₂ — Fit interaction models inside each cluster (i.e., per-cluster model that also uses interaction terms).

2. The central idea (informal)

If the data-generating process includes a strong interaction term 
𝛽
𝑖
𝑗
(
𝑋
𝑖
𝑋
𝑗
)
β
ij
	​

(X
i
	​

X
j
	​

), then for fixed 
𝑋
𝑗
=
𝑐
X
j
	​

=c we have

𝐸
[
𝑌
∣
𝑋
𝑖
,
𝑋
𝑗
=
𝑐
]
≈
𝛽
𝑖
𝑗
𝑐
⋅
𝑋
𝑖
+
(small terms)
.
E[Y∣X
i
	​

,X
j
	​

=c]≈β
ij
	​

c⋅X
i
	​

+(small terms).

So conditioning on 
𝑋
𝑗
X
j
	​

 transforms the multiplicative effect into a linear relationship in 
𝑋
𝑖
X
i
	​

 with slope proportional to 
𝑐
c. If 
𝑋
𝑗
X
j
	​

 takes different typical values in different regions (e.g., positive vs negative, low vs high), these regions are natural clusters where the 
𝑋
𝑖
X
i
	​

-to-
𝑌
Y relationship is approximately linear but with different slopes. Clustering along 
𝑋
𝑗
X
j
	​

 therefore turns a global nonlinear interaction into locally linear relationships, enabling simple models to perform much better.

3. Proof 1 — Piecewise linear / conditional-slope argument
Statement

Given

𝑌
=
𝛽
0
+
𝛽
𝑖
𝑋
𝑖
+
𝛽
𝑗
𝑋
𝑗
+
𝛽
𝑖
𝑗
(
𝑋
𝑖
𝑋
𝑗
)
+
𝜀
,
Y=β
0
	​

+β
i
	​

X
i
	​

+β
j
	​

X
j
	​

+β
ij
	​

(X
i
	​

X
j
	​

)+ε,

the conditional slope of 
𝑌
Y w.r.t. 
𝑋
𝑖
X
i
	​

 is

∂
𝐸
[
𝑌
∣
𝑋
𝑗
]
∂
𝑋
𝑖
=
𝛽
𝑖
+
𝛽
𝑖
𝑗
𝑋
𝑗
.
∂X
i
	​

∂E[Y∣X
j
	​

]
	​

=β
i
	​

+β
ij
	​

X
j
	​

.

Hence, fixing 
𝑋
𝑗
X
j
	​

 (or splitting 
𝑋
𝑗
X
j
	​

 into regions) gives different linear relationships in 
𝑋
𝑖
X
i
	​

. If 
∣
𝛽
𝑖
𝑗
∣
∣β
ij
	​

∣ is large relative to 
∣
𝛽
𝑖
∣
∣β
i
	​

∣, then the slope sign/direction changes with 
𝑋
𝑗
X
j
	​

.

Consequence

Split data by threshold(s) on 
𝑋
𝑗
X
j
	​

 (e.g., sign or median), fit separate linear models in each split → those piecewise linear models capture most of the interaction effect.

4. Proof 2 — Geometric interpretation (surface twisting)
Statement

Consider 
(
𝑋
𝑖
,
𝑋
𝑗
,
𝑌
)
(X
i
	​

,X
j
	​

,Y) space. Without the interaction term the expected response surface is planar:

𝑌
=
𝛽
0
+
𝛽
𝑖
𝑋
𝑖
+
𝛽
𝑗
𝑋
𝑗
.
Y=β
0
	​

+β
i
	​

X
i
	​

+β
j
	​

X
j
	​

.

Adding 
𝛽
𝑖
𝑗
𝑋
𝑖
𝑋
𝑗
β
ij
	​

X
i
	​

X
j
	​

 produces a saddle-like or twisted surface:

𝑌
=
𝛽
0
+
𝛽
𝑖
𝑋
𝑖
+
𝛽
𝑗
𝑋
𝑗
+
𝛽
𝑖
𝑗
𝑋
𝑖
𝑋
𝑗
.
Y=β
0
	​

+β
i
	​

X
i
	​

+β
j
	​

X
j
	​

+β
ij
	​

X
i
	​

X
j
	​

.

The sign of 
𝑋
𝑖
𝑋
𝑗
X
i
	​

X
j
	​

 partitions the domain into different lobes of the surface, which correspond to distinct regimes. Each lobe can be approximated well by a local plane (a cluster-specific linear model).

5. Proof 3 — Variance decomposition & residual conditioning
Statement

Total variance:

V
a
r
(
𝑌
)
=
V
a
r
(
𝐸
[
𝑌
∣
𝐶
]
)
+
𝐸
[
V
a
r
(
𝑌
∣
𝐶
)
]
Var(Y)=Var(E[Y∣C])+E[Var(Y∣C)]

If we set 
𝐶
C as a clustering indicator (e.g., sign of 
𝑋
𝑗
X
j
	​

), and the dominant portion of 
V
a
r
(
𝑌
)
Var(Y) arises from the interaction 
𝛽
𝑖
𝑗
𝑋
𝑖
𝑋
𝑗
β
ij
	​

X
i
	​

X
j
	​

, then 
V
a
r
(
𝐸
[
𝑌
∣
𝐶
]
)
Var(E[Y∣C]) will be large, and within-cluster residual variance will be comparatively small. Therefore clustering explains a large share of variance and improves 
𝑅
2
R
2
.

6. Expected 
𝑅
2
R
2
 improvement from clustering (intuition & formula)

If for cluster 
𝑘
k we have (approximately) a linear model with variance explained 
𝑅
𝑘
2
R
k
2
	​

, and cluster weight 
𝑤
𝑘
w
k
	​

, the combined clustered 
𝑅
2
R
2
 is

𝑅
clustered
2
=
∑
𝑘
𝑤
𝑘
𝑅
𝑘
2
.
R
clustered
2
	​

=
k
∑
	​

w
k
	​

R
k
2
	​

.

If the interaction explains variance proportional to 
𝛽
𝑖
𝑗
2
⋅
V
a
r
(
𝑋
𝑖
𝑋
𝑗
)
β
ij
2
	​

⋅Var(X
i
	​

X
j
	​

), then clustering that isolates directions where 
𝑋
𝑗
X
j
	​

 (or 
𝑋
𝑖
X
i
	​

) takes distinct values converts multiplicative variance into linear explained variance, increasing per-cluster 
𝑅
2
R
2
. The expected improvement scales with 
𝛽
𝑖
𝑗
2
⋅
V
a
r
(
𝑋
𝑖
𝑋
𝑗
)
β
ij
2
	​

⋅Var(X
i
	​

X
j
	​

) and with how well the cluster rule separates distinct conditional means.

7. From theory to practice — algorithmic IBC pipeline

A compact practical pipeline:

Feature engineering: standardize numeric variables; create candidate interactions up to desired degree (use interaction_only=True to avoid powers if desired).

Screen / select interactions: use LassoCV / ElasticNetCV on standardized interaction features to find top interactions.

Select clustering variable(s): choose one variable from the top interaction (e.g., for 
𝑋
𝑖
𝑋
𝑗
X
i
	​

X
j
	​

 choose 
𝑋
𝑗
X
j
	​

, or evaluate both). Optionally consider joint clustering if interaction is multi-variable.

Define split rule: sign/median/quantile/tree split on selected variable(s).

Fit per-cluster models: simple linear regression or models that include interaction features.

Evaluate: compare A (no interaction global), C (global with interaction), B (clustered simple), D (clustered models including interactions) via cross-validated metrics.

8. Simulations and experiments (reproducible code)

Below are compact, runnable code snippets demonstrating the key concepts. Replace variables as needed.

Note: run these blocks in a Python environment with numpy, pandas, scikit-learn, matplotlib.

8.1 Generate a dataset where interaction 
𝑋
𝑖
𝑋
𝑗
X
i
	​

X
j
	​

 is dominant
import numpy as np, pandas as pd
np.random.seed(42)
n = 1000
Xi = np.random.normal(0,1,n)
Xj = np.random.normal(0,1,n)
beta0, betai, betaj, betaij = 0.5, 0.1, 0.1, 3.0
Y = beta0 + betai*Xi + betaj*Xj + betaij*(Xi*Xj) + np.random.normal(0,1,n)
df = pd.DataFrame({'Xi': Xi, 'Xj': Xj, 'Y': Y})

8.2 Compare models A, B (IBC), C
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# A: global linear without interaction
X_A = df[['Xi','Xj']]
y = df['Y'].values
modelA = LinearRegression().fit(X_A, y)
r2A = r2_score(y, modelA.predict(X_A))

# B: IBC - split by Xj >= 0
mask = df['Xj'] >= 0
preds = np.empty(len(df))
model1 = LinearRegression().fit(X_A[mask], y[mask])
preds[mask] = model1.predict(X_A[mask])
model2 = LinearRegression().fit(X_A[~mask], y[~mask])
preds[~mask] = model2.predict(X_A[~mask])
r2B = r2_score(y, preds)

# C: global linear with interaction
X_C = np.column_stack([df['Xi'], df['Xj'], df['Xi']*df['Xj']])
modelC = LinearRegression().fit(X_C, y)
r2C = r2_score(y, modelC.predict(X_C))

print("R² A (no interaction):", r2A)
print("R² B (IBC clustered):", r2B)
print("R² C (with interaction):", r2C)

8.3 D₁/D₂: Fit interaction models inside clusters (possible improvement over C)
# D models (per-cluster interaction models)
mask = df['Xj'] >= 0
# D1
X1 = np.column_stack([df.loc[mask,'Xi'], df.loc[mask,'Xj'], df.loc[mask,'Xi']*df.loc[mask,'Xj']])
y1 = df.loc[mask, 'Y'].values
mD1 = LinearRegression().fit(X1, y1)
# D2
X2 = np.column_stack([df.loc[~mask,'Xi'], df.loc[~mask,'Xj'], df.loc[~mask,'Xi']*df.loc[~mask,'Xj']])
y2 = df.loc[~mask, 'Y'].values
mD2 = LinearRegression().fit(X2, y2)

predsD = np.empty(len(df))
predsD[mask] = mD1.predict(X1)
predsD[~mask] = mD2.predict(X2)
r2D = r2_score(y, predsD)
print("R² D (clustered interaction models):", r2D)

9. When D₁/D₂ can beat global interaction model C

D₁/D₂ can outperform C when:

Regime-dependent coefficients: 
𝛽
𝑖
𝑗
β
ij
	​

 (or other coefficients) differ significantly between clusters.

Regime-specific intercepts or omitted variables: each regime has unique baseline or unobserved features.

Heteroskedasticity: different noise levels across regimes.

Nonlinearities only active in certain regimes: global polynomial fails to capture localized nonlinearity.

Finite-sample / regularization effects: splitting reduces variance of parameter estimates relative to a single global model that over-regularizes.

Diagnosis: fit C and D on held-out data (or CV). If D has higher out-of-sample 
𝑅
2
R
2
 or lower error, the data are heterogeneous and regime models are warranted.

10. High-dimensional & real-world considerations

When 
𝑝
p is large (genomics, finance, ads):

Feature explosion: number of interactions grows combinatorially. Use screening (univariate filters, domain priors) to limit candidates.

Regularization: use LassoCV, ElasticNetCV, or hierarchical/group penalties (group Lasso) to select interaction groups.

Clustering rule complexity: interactions of degree >2 often require multivariate cluster rules (small decision trees, k-means, mixture models) rather than a simple single-variable sign split.

Stability & interpretability: trees or sparse methods help find stable splits and interpretable clusters.

Cross-validation: essential to validate whether clustered models generalize better.

Domain-specific notes:

Finance: cluster on volatility quantiles or regime indicators discovered by interactions of macro variables.

Genomics: screen SNPs/genes by marginal tests, then investigate pairwise/triple interactions for epistasis; cluster on gene expression patterns.

Marketing / Social: interactions between channel × demographic × time can reveal segments for targeted models.

11. Practical recommendations & diagnostics

Start small: try degree-2 interactions first. Use interaction_only=True.

Standardize raw features before building polynomial features (important for Lasso).

Screen candidates (univariate correlations, domain knowledge) to reduce search.

When top interaction 
𝑋
𝑖
𝑋
𝑗
X
i
	​

X
j
	​

 found, visualize 
𝑋
𝑖
X
i
	​

 vs 
𝑌
Y colored by 
𝑋
𝑗
X
j
	​

 (or vice versa). If branches appear, IBC likely helps.

Compare models A/B/C/D on held-out sets (not just in-sample R²).

If D wins but cluster sizes are small, consider hierarchical or mixture-of-experts models to borrow strength.

Automate selection of the clustering variable by scanning candidate variables in top interactions and comparing cross-validated gains in 
𝑅
2
R
2
. (Code snippets earlier show how.)

12. Conclusion & future directions

IBC connects model interpretability, feature selection, and clustering: important interaction features reveal regime structure; splitting on those regimes often converts a global nonlinear relationship into locally linear pieces that are easier to model and interpret. Key takeaways:

If interaction truly generates the signal and is homogeneous, model C (global interaction) is optimal.

If interaction behavior or noise differs by regime, cluster-specific interaction models (D) can beat global models.

In high-dimensional settings, combine screening + sparse selection + careful cross-validation.

Future work: rigorous statistical tests for when to prefer D over C (penalized likelihood criteria, hierarchical Bayes), automatic multi-variable split discovery, and scalable IBC for ultra-high-dimensional data.

13. Appendix — compact code summary

Below is a compact snippet that performs the full automated IBC discovery (selection → cluster → compare models). It is a simplified blueprint — tune and extend for production.

# Compact IBC pipeline (simplified)
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score

def ibc_pipeline(df, target, degree=2, top_k=5):
    X = df.drop(columns=[target])
    y = df[target].values
    # 1. Create interaction-only features and names
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    Xpoly = poly.fit_transform(X)
    names = poly.get_feature_names_out(X.columns)
    Xpoly_df = pd.DataFrame(Xpoly, columns=names)
    # 2. Standardize and LassoCV
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xpoly_df)
    lasso = LassoCV(cv=5, random_state=0).fit(Xs, y)
    coefs = pd.Series(np.abs(lasso.coef_), index=names).sort_values(ascending=False)
    top = coefs.head(top_k).index.tolist()
    print("Top interactions:", top)
    # 3. Choose variable from the top interaction (last token)
    candidate = top[0].split()[-1]  # crude choice
    print("Clustering by:", candidate)
    mask = df[candidate] >= df[candidate].median()
    # 4. Fit clustered linear models (no interactions)
    model1 = LinearRegression().fit(X[mask], y[mask])
    model2 = LinearRegression().fit(X[~mask], y[~mask])
    preds = np.empty(len(y))
    preds[mask] = model1.predict(X[mask])
    preds[~mask] = model2.predict(X[~mask])
    r2_clustered = r2_score(y, preds)
    # 5. Fit global interaction model (use Xpoly_df)
    modelC = LinearRegression().fit(Xpoly_df, y)
    r2_global = r2_score(y, modelC.predict(Xpoly_df))
    return {'r2_clustered': r2_clustered, 'r2_global_interaction': r2_global, 'top_interactions': top}

# Usage:
# result = ibc_pipeline(df, 'Y', degree=2)
# print(result)