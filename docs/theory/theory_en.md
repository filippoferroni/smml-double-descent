# Double descent in linear models: theory (Assignment 4, SMML 2025/26)

> Version 4, 25/10/2026. Typeset version: `theory_en.pdf`. Versione italiana: `teoria_it.md`.

#### How to use this document.

Reading time is about 40 minutes. Sections 2–6 are the minimum theory to be able to explain at the oral exam. Section 7 links the theory to the gradient descent extension, Section 8 describes what we expect to see in the plots, Section 9 the experimental design. At the end: likely oral questions with answers, and solutions to the check questions. Every claim states its source; when a concept is **not** in the uploaded course materials (Cesa-Bianchi’s notes 1–15) it is flagged explicitly as an external source.

## Notation and correspondences

<div class="center">

| Concept                         | Course (slides 9)              | Assignment / paper             | Code    |
|:--------------------------------|:-------------------------------|:-------------------------------|:--------|
| number of training examples     | $m$                            | $n$                            | `m`     |
| number of features (complexity) | $d$                            | $d$ (assignment) / $N$ (paper) | `d`     |
| design matrix                   | $X \in \mathbb{R}^{m\times d}$ |                                | `X`     |
| labels                          | $y \in \mathbb{R}^m$           |                                | `y`     |
| ridge parameter                 | $\alpha$                       | $\lambda$ (other texts)        | `alpha` |
| interpolation threshold         | $d = m$                        | $N = n$                        |         |

</div>

**Careful:** in the paper $d$ is the dimension of the *input* $x$, while the number of features/parameters is $N$ and the number of samples is $n$ (p. 2 and p. 4).

In the report we state once: “we follow the course notation; the number of examples $n$ in the assignment corresponds to $m$”. The rows of $X$ are the examples: $X^\top = [x_1,\dots,x_m]$ (slides 9, p. 5). A linear predictor for regression is $h(x) = w^\top x$ with $w \in \mathbb{R}^d$ (slides 9, p. 1 and p. 4). The intercept is handled by adding a constant feature equal to $1$ (slides 9, p. 1) or by centering the data.

## Statistical setup: risk, training error, test error

*Source: slides “4. statRisk” (p. 1–4) and “1. intro” (p. 3–5).*

- Every example $(x,y)$ is an independent draw from a fixed but unknown distribution $\mathcal{D}$ (statRisk, p. 1). Training and test sets are therefore statistical samples.

- **Statistical risk** of a predictor $h$ with respect to the loss $\ell$ (statRisk, p. 1): $\ell_{\mathcal{D}}(h) = \mathbb{E}\bigl[\ell(h(X), Y)\bigr]$. For us $\ell(\hat y, y) = (\hat y - y)^2$ and $h(x) = w^\top x$.

- **Bayes optimal predictor** $f^\ast$: the best possible one given $\mathcal{D}$. For the square loss $f^\ast(x) = \mathbb{E}[Y \mid X = x]$ and the **Bayes risk** is the expected conditional variance of the label (statRisk, p. 1–2). In our synthetic dataset, with $y = w^{\star\top}x + \varepsilon$ and noise variance $\sigma^2$, the Bayes risk is exactly $\sigma^2$: **no model can have expected test error below $\sigma^2$**. We will use it as a reference line in the plots.

- **Training error**: $\ell_S(h) = \frac{1}{m}\sum_{t=1}^{m} \ell(h(x_t), y_t)$; for us $\ell_S(w) = \frac{1}{m}\left\lVert Xw - y \right\rVert^2$. ERM picks the predictor minimizing it (intro, p. 4).

- **Test error**: the same average on an independent test set $S'$; it estimates the risk $\ell_{\mathcal{D}}(h)$, and by Chernoff–Hoeffding the estimate is accurate when the test set is large (statRisk, p. 3–4). The test set is never used for training or for choosing hyperparameters.

#### Overfitting and underfitting in the course.

Intro, p. 5: if the algorithm returns a predictor with high training error it is *underfitting*; with low training error but high test error it is *overfitting*. The course notes that overfitting is expected when the predictor class is too rich compared to $m$ (“$\log_2|F| \gg m$”, intro p. 5). **Double descent is precisely a case where this classical intuition is not enough**: for $d \gg m$ the training error is $0$ but the test error goes down again.

## Least squares with $d \le m$: the formula in the slides

*Source: slides 9, p. 4–5.*

The linear regression predictor is the ERM with respect to the square loss: $$w_S = \operatorname*{argmin}_{w \in \mathbb{R}^d} \left\lVert Xw - y \right\rVert^2 .$$ $F(w) = \left\lVert Xw - y \right\rVert^2$ is convex, so the minimum is where the gradient vanishes: $$\nabla F(w) = 2X^\top(Xw - y) = 0
  \quad\Longleftrightarrow\quad
  X^\top X\, w = X^\top y \qquad \text{(normal equations)}.$$ If $X^\top X$ is invertible: $$\boxed{w_S = (X^\top X)^{-1} X^\top y}$$ **When is it invertible?** The slides say: if and only if $x_1,\dots,x_m$ span $\mathbb{R}^d$. At least $d$ vectors are needed to span $\mathbb{R}^d$, so we need $m \ge d$.

#### Geometric interpretation.

The vector of training predictions is $Xw_S = X(X^\top X)^{-1}X^\top y = Py$, where $P$ is the orthogonal projection onto the column space of $X$. The more features we add, the larger that space and the closer $Py$ gets to $y$: the training error **cannot increase** as $d$ grows.

#### The case $d = m$.

If $X$ is square and invertible, $w_S = X^{-1}y$ and $Xw_S = y$: training error exactly $0$. The model already **interpolates** at $d = m$. Hence the name “interpolation threshold”.

## Least squares with $d > m$: the minimum-norm solution

*External source (not in the course materials): Moore–Penrose pseudo-inverse, standard linear algebra textbooks (e.g. Strang, *Introduction to Linear Algebra*, chapter on the pseudo-inverse and SVD); use of the minimum-norm solution: Belkin et al. 2019, Sec. 2 (p. 4: “when the minimizer is not unique … we choose the minimizer whose coefficients have the minimum $\ell_2$ norm”) and p. 10 (“minimum norm linear regression in the feature space”).*

## Why the slides’ formula no longer works

$X^\top X$ is a $d \times d$ matrix, but its rank equals the rank of $X$, which is at most $\min(m,d) = m$. If $d > m$ then $\operatorname{rank}(X^\top X) \le m < d$: the matrix is **singular** and $(X^\top X)^{-1}$ does not exist. Equivalently: $m$ vectors cannot span $\mathbb{R}^d$ when $m < d$.

## Infinitely many interpolating solutions

With $d > m$ (and linearly independent rows of $X$, the typical case with continuous data), the system $Xw = y$ has $m$ equations and $d$ unknowns: it has **infinitely many** exact solutions. If $w_0$ is a solution, so is $w_0 + z$ for every $z$ in the null space of $X$ ($Xz = 0$), which has dimension $d - m$. All of them have zero training error.

## Which one to pick: the minimum-norm solution

$$\boxed{w^{+} = \operatorname*{argmin}\bigl\{\left\lVert w \right\rVert : Xw = y\bigr\} = X^\top (X X^\top)^{-1} y}$$ **Check that it interpolates:** $Xw^{+} = XX^\top(XX^\top)^{-1}y = y$. ($XX^\top$ is $m\times m$ and invertible when the rows of $X$ are independent.)

**Why it has minimum norm (short proof, know how to do it).** $w^{+} = X^\top a$ with $a = (XX^\top)^{-1}y$, so $w^{+}$ is a linear combination of the rows of $X$. Any other solution is $w = w^{+} + z$ with $Xz = 0$, i.e. $z$ is orthogonal to every row of $X$, hence to $w^{+}$. By Pythagoras: $$\left\lVert w \right\rVert^2 = \left\lVert w^{+} \right\rVert^2 + \left\lVert z \right\rVert^2 \ge \left\lVert w^{+} \right\rVert^2 .$$

## One formula for everything: the pseudo-inverse

The pseudo-inverse $X^{+}$ is computed from the SVD $X = U\Sigma V^\top$: $X^{+} = V\Sigma^{+}U^\top$, where $\Sigma^{+}$ inverts the non-zero singular values and leaves the others at zero. Then:

- if $d \le m$ and $X$ has full rank: $X^{+}y = (X^\top X)^{-1}X^\top y = w_S$ (the slides’ formula);

- if $d > m$ and $X$ has rank $m$: $X^{+}y = X^\top(XX^\top)^{-1}y = w^{+}$.

So in numpy `w = np.linalg.pinv(X) @ y` works for **every** $d$. It is linear algebra, not a library “fit”, so it respects the “from scratch” rule (assignment FAQ, p. 16).

#### Numerical notes (important for the code).

- Do not use `np.linalg.inv(X.T @ X)`: forming $X^\top X$ squares the condition number, and near $d=m$ the results become numerically unreliable.

- `np.linalg.pinv` only zeroes singular values below a tiny threshold (`rcond`, about $10^{-15}$ relative to the largest). Near $d=m$ the small singular values are *not* zeroed: `pinv` does not regularize, so the peak stays visible. This is the behaviour we want to show.

## Minimal numerical example

$m=1$, $d=2$: a single example $x = (1,1)$, $y = 2$. The solutions of $w_1 + w_2 = 2$ are infinitely many: $(2,0)$, $(0,2)$, $(1,1)$, $(3,-1)$… The minimum-norm solution is $w^{+} = x^\top(xx^\top)^{-1}y = (1,1)\cdot\frac{2}{2} = (1,1)$, with norm $\sqrt{2}$, versus norm $2$ for $(2,0)$. It spreads the weight evenly instead of “betting everything” on one feature.

## Why the test error blows up at $d \approx m$

*Source: slides 9, p. 5: “when $X^\top X$ is nearly singular, $w_S$ is highly sensitive to perturbations of the training set. This instability increases the estimation error (or variance)”. This is the hook to the course.*

## Intuition with a $2\times 2$ example

$m = d = 2$, rows $x_1 = (1,0)$, $x_2 = (1,\varepsilon)$ with small $\varepsilon$ (almost parallel rows, $X$ nearly singular). Labels $y = (1,\, 1+\delta)$, where $\delta$ is a small noise. Solving $Xw = y$: $w_1 = 1$ and $w_1 + \varepsilon w_2 = 1+\delta$, so $$w_2 = \delta/\varepsilon .$$ With $\varepsilon = 0.01$ and noise $\delta = 0.1$ we get $w_2 = 10$: a small noise produces a huge weight. On a new point, e.g. $x = (0,1)$, the prediction is $10$ instead of roughly $0$. That is exactly the peak.

## General version with eigenvalues

Let $\lambda_1 \ge \dots \ge \lambda_d \ge 0$ be the eigenvalues of $X^\top X$ and $v_1,\dots,v_d$ the (orthonormal) eigenvectors. The minimum-norm solution reads $$w^{+} = \sum_{i:\,\lambda_i > 0} \frac{v_i^\top X^\top y}{\lambda_i}\, v_i .$$ Assume noisy linear labels: $y = Xw^\star + \varepsilon$. The noise part is $\sum_i \frac{v_i^\top X^\top \varepsilon}{\lambda_i} v_i$; each term has norm $\frac{|u_i^\top \varepsilon|}{\sqrt{\lambda_i}}$, with $u_i = Xv_i/\sqrt{\lambda_i}$ a unit vector. **The noise is divided by $\sqrt{\lambda_i}$.**

- $d \ll m$: $X$ is “tall and thin”, all eigenvalues are large (about $m$ for standardized features): the noise is damped.

- $d \approx m$: $X$ is nearly square and random, the smallest eigenvalue is very close to $0$: the noise is amplified, $\left\lVert w \right\rVert$ blows up, and so does the test error.

- $d \gg m$: $X$ is “wide”, the $m$ non-zero eigenvalues move away from $0$ again and the noise is spread over many directions: $\left\lVert w^{+} \right\rVert$ and the test error decrease. This is the **second descent**.

*External source (not in the course materials): the behaviour of the smallest eigenvalue of a random matrix as a function of $d/m$ is a random matrix theory result (Marchenko–Pastur law). We do not prove it: we show it empirically by plotting $\left\lVert w \right\rVert$ against $d$, which is easier to defend at the oral.*

## An exact formula for the classical regime (optional)

*External source (not in the course materials): Hastie, Montanari, Rosset, Tibshirani, “Surprises in high-dimensional ridgeless least squares interpolation”, Annals of Statistics 2022.*

If $x \sim \mathcal{N}(0, I_d)$, $y = w^{\star\top}x + \varepsilon$ with noise variance $\sigma^2$ and $d < m-1$, then $$\mathbb{E}\bigl[\left\lVert w_S - w^\star \right\rVert^2\bigr] = \sigma^2\,\frac{d}{m - d - 1},$$ which **diverges as $d \to m-1$**. No need to prove it; it is useful to say at the oral that the peak is not a numerical artefact but a statistical phenomenon predicted by theory.

## Ridge regression

*Source: slides 9, p. 5.*

## Formula

$$w_{S,\alpha} = \operatorname*{argmin}_{w \in \mathbb{R}^d} \left\lVert Xw - y \right\rVert^2 + \alpha \left\lVert w \right\rVert^2 ,
  \qquad \alpha > 0 .$$ Gradient: $2X^\top(Xw - y) + 2\alpha w = 0 \;\Rightarrow\; (X^\top X + \alpha I)\,w = X^\top y$, hence $$\boxed{w_{S,\alpha} = (\alpha I + X^\top X)^{-1} X^\top y}$$

## Why it removes the peak

The eigenvalues of $\alpha I + X^\top X$ are $\alpha + \lambda_i \ge \alpha > 0$ (slides 9, p. 5): the matrix is always invertible, for every $d$, including $d > m$. In the eigenvector basis: $$w_{S,\alpha} = \sum_{i} \frac{v_i^\top X^\top y}{\lambda_i + \alpha}\, v_i .$$ We divide by $\lambda_i + \alpha$ instead of $\lambda_i$: even when $\lambda_i \approx 0$ the denominator stays at least $\alpha$, so the noise cannot be amplified without bound. The peak is damped or disappears.

## The price: bias

- $\alpha \to 0$: we recover the minimum-norm solution $w^{+}$ (also for $d > m$; along directions with $\lambda_i = 0$ the numerator $v_i^\top X^\top y$ is $0$). *Standard fact, not in the slides.*

- $\alpha \to \infty$: $w_{S,\alpha} \to 0$, the predictor ignores the data (slides 9, p. 5: $\alpha$ controls the bias).

- In between: $\alpha$ trades variance (instability) for bias (underfitting). With too large an $\alpha$ the test error goes up *for every* $d$.

*Source: Belkin et al. 2019, p. 10: regularization “can both prevent interpolation and change the effective capacity of the function class, thus attenuating or masking the interpolation peak”.*

*External source (not in the course materials): Nakkiran, Venkat, Kakade, Ma, “Optimal regularization can mitigate double descent”, ICLR 2021: with the optimal $\alpha$ the test error curve becomes monotone.*

## Computing it in practice

- Primal: `np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ y)` ($d \times d$ system).

- Dual, cheaper for $d > m$: $w_{S,\alpha} = X^\top(XX^\top + \alpha I)^{-1}y$ ($m\times m$ system). The two forms agree because $X^\top(XX^\top + \alpha I) = (X^\top X + \alpha I)X^\top$.

- `solve` is preferable to computing the explicit inverse (more stable and faster).

## Choosing $\alpha$ without data leakage

- For the main plot: a few curves with fixed $\alpha$ (e.g. $10^{-3}, 10^{-1}, 10$) next to the minimum-norm curve. No data-driven choice, no leakage.

- Time permitting: $\alpha$ chosen for each $d$ on a validation split (or $K$-fold cross-validation) taken **only** from the training set (slides “6. crossVal”, p. 2–3: tuning on a validation set; p. 3–4: nested CV), as in the professor’s `CrossValidation.ipynb` notebook. The test set is never used for selection.

## Link with gradient descent (extension)

*Source: slides 10 (online gradient descent) for the algorithm; the minimum-norm result is external (see below).*

Gradient descent on $F(w) = \left\lVert Xw - y \right\rVert^2$ started at $w_1 = 0$: $$w_{k+1} = w_k - \eta\, 2X^\top(Xw_k - y).$$

- Every update is $X^\top(\dots)$, a linear combination of the rows of $X$. Starting from $0$, $w_k$ always stays in the row space.

- With a small enough step size ($\eta < 1/\lambda_1$), for $d > m$ GD converges to a solution that interpolates *and* lies in the row space: by the proof in Section 4.3, this is exactly $w^{+}$.

- Stopping before convergence (early stopping) has an effect similar to ridge: directions with small $\lambda_i$ are learned last, so the noise along them is not amplified.

*Source: implicit bias of GD towards minimum norm: Belkin et al. 2019, p. 11 (“SGD initialized at zero also converges to the minimum norm solution”).*

The extension is done **only** once the report and repo are complete.

## What we expect in the plots

## The classical U-curve and double descent

- **Classical regime** ($d < m$): the course decomposes the risk of $h_S$ into *estimation error* (variance) $+$ *approximation error* (bias) $+$ Bayes risk (statRisk, p. 4). Adding features lowers the bias but raises the variance; overfitting when variance dominates, underfitting when bias dominates (statRisk, p. 6). Near $d = m$ the variance diverges: this is the U-curve.

- **Interpolation threshold** ($d = m$): training error reaches $0$, test error peaks.

- **Over-parameterized regime** ($d > m$): training error is always $0$, the test error decreases again and can go below the classical minimum (paper, Fig. 1, p. 3). The paper’s explanation (p. 3 and p. 6): beyond the threshold, richer classes contain interpolants with smaller norm, i.e. “simpler” ones (Occam’s razor); the norm peaks at the threshold and then decreases (Fig. 2, p. 5). Classical theory, which predicts only the U, does not explain this part: this is the point of the paper.

## Planned plots

1.  Training and test error against $d$ (minimum norm), vertical line at $d = m$, logarithmic $y$ axis (the peak spans orders of magnitude).

2.  On the synthetic data, a horizontal line at the Bayes risk $\sigma^2$ (Section 2) as a reference.

3.  $\left\lVert w \right\rVert$ against $d$: it must peak at the same place (evidence for the mechanism of Section 5).

4.  Test error of minimum norm and ridge (several $\alpha$) on the same plot.

5.  The same plots for the real dataset, and a comparison between the two datasets.

## Experimental design (preview)

#### Relation to the paper.

The paper shows the phenomenon with random Fourier and random ReLU features (Sec. 2 and App. C.1–C.2), neural networks (Sec. 2) and forests/boosting (Sec. 3), on MNIST and other classification datasets. It has no “pure” linear regression experiment, but the random features are trained by minimum-norm linear regression in feature space (p. 10): our experiment is its simplest version.

## Synthetic dataset: noisy linear labels

*External source (not in the course materials): “weak features” model: Belkin, Hsu, Xu, “Two models of double descent for weak features”, 2019.*

- $x \sim \mathcal{N}(0, I_D)$ with large $D$ (e.g. $D = 1000$), $y = w^{\star\top}x + \varepsilon$, Gaussian noise.

- The model with complexity $d$ uses only the first $d$ features. For $d < D$ the unused features act as extra noise (bias); for large $d$ the model has access to all the signal.

- Fixed $m$ (e.g. $m = 100$), $d$ from $1$ to $D$, denser grid around $d = m$.

- Mean (or median) over several repetitions with different seeds, with a variability band.

## Real dataset with random features

- A regression dataset with few features (e.g. California Housing, 8 features, also in the professor’s repo).

- To push $d$ past $m$: random features $\phi(x) = \max(0, Wx)$ (ReLU) with a fixed Gaussian $W$, or random Fourier features as in the paper. The model stays **linear in the parameters** $w$. The course calls this idea *feature expansion*: adding new features built from the original ones to reduce the bias of linear predictors (slides “11. kernels”, p. 1).

- We subsample $m$ training examples (e.g. $m = 200$) and keep a separate test set.

## Rules against data leakage

- Train/test split *before* any pre-processing.

- Mean and standard deviation for standardization computed **only** on the training set and applied to the test set.

- The random matrices $W$ do not depend on the data, so they create no leakage, but they must be generated with a fixed seed.

- Hyperparameters (e.g. $\alpha$) chosen only with training data.

## Likely oral questions, with short answers

1.  **What is double descent?** As a function of complexity, the test error decreases, rises to a peak at the interpolation threshold ($d=m$), then decreases again in the over-parameterized regime.

2.  **Why is the peak exactly at $d = m$?** There $X$ is nearly square, $X^\top X$ has an almost-zero eigenvalue, the solution divides the noise by that eigenvalue and $\left\lVert w \right\rVert$ blows up (slides 9, p. 5 and Section 5).

3.  **Why is the training error $0$ for $d \ge m$?** With $d \ge m$ and independent rows, $Xw = y$ has at least one exact solution: the model interpolates.

4.  **Which solution do you use for $d > m$ and why?** The minimum-norm $X^{+}y$: it is unique, interpolates, is the limit of ridge as $\alpha \to 0$ and the point GD converges to when started at $0$.

5.  **How did you compute $X^{+}$?** With `np.linalg.pinv`, based on the SVD. It is linear algebra, not a fit.

6.  **What does ridge do to the peak?** It replaces $\lambda_i$ with $\lambda_i + \alpha$ in the denominator: no noise amplification, peak damped or removed, at the cost of some bias.

7.  **How did you choose $\alpha$?** Fixed values compared side by side, and/or validation on the training set only. Never on the test set.

8.  **How do you vary $d$ on the real dataset?** With random features; the model stays linear in the parameters.

9.  **How do you avoid data leakage?** Split first, standardization statistics from the training set only, hyperparameters from the training set only.

10. **Why does double descent contradict classical theory?** Classical theory predicts that beyond some point more complexity always hurts generalization (U-curve). In the $d > m$ regime the opposite happens.

11. **Weak points of your work?** Linear models only; the real dataset needs random features, so the result depends on how they are built; few repetitions because of time constraints; arbitrary choice of $m$.

## Solutions to the check questions

*Try to answer on your own first, then compare.*

#### Question 1.

Why can’t the slides’ formula $(X^\top X)^{-1}X^\top y$ be used for $d > m$, and what does `pinv(X) @ y` compute?

*Answer.* $X^\top X$ is $d\times d$ but has rank at most $m < d$, so it is singular and not invertible (equivalently, $m$ vectors cannot span $\mathbb{R}^d$, slides 9 p. 4). The system $Xw = y$ has infinitely many interpolating solutions. `pinv(X) @ y` returns the one with minimum Euclidean norm, $X^\top(XX^\top)^{-1}y$: it is unique, lies in the row space of $X$, and every other solution has a larger norm (Pythagoras). For $d \le m$ it coincides with the slides’ formula.

#### Question 2.

Why does ridge have no peak at $d = m$?

*Answer.* In the minimum-norm solution the components of $y$ are divided by the eigenvalues $\lambda_i$ of $X^\top X$; at $d \approx m$ the smallest one is almost $0$ and the noise blows up. Ridge inverts $\alpha I + X^\top X$, whose eigenvalues are $\lambda_i + \alpha \ge \alpha > 0$ (slides 9 p. 5): the denominator never drops below $\alpha$, so $\left\lVert w \right\rVert$ and the test error stay bounded. The price is bias: with too large an $\alpha$, $w \to 0$ and the model underfits.

## Sources

- N. Cesa-Bianchi, notes “Introduction” (1. intro), p. 3–5; “Statistical learning” (4. statRisk), p. 1–6; “Hyperparameter tuning and risk estimates” (6. crossVal), p. 2–4; “Kernels” (11. kernels), p. 1.

- N. Cesa-Bianchi, slides “Linear predictors” (9. linear), version of 10/02/2026, p. 4–5.

- N. Cesa-Bianchi, slides “Online Gradient Descent” (10. online), version of 24/02/2026.

- M. Belkin, D. Hsu, S. Ma, S. Mandal, “Reconciling modern machine learning practice and the classical bias-variance trade-off”, PNAS 2019 (arXiv:1812.11118). arXiv v2 (10/09/2019); page numbers refer to that PDF.

- M. Belkin, D. Hsu, J. Xu, “Two models of double descent for weak features”, 2019 (arXiv:1903.07571). *External.*

- T. Hastie, A. Montanari, S. Rosset, R. Tibshirani, “Surprises in high-dimensional ridgeless least squares interpolation”, Annals of Statistics 2022. *External.*

- P. Nakkiran, P. Venkat, S. Kakade, T. Ma, “Optimal regularization can mitigate double descent”, ICLR 2021. *External.*

- Assignment “ML projects 25/26”, Assignment 4 (p. 8–9) and FAQ (p. 16).
