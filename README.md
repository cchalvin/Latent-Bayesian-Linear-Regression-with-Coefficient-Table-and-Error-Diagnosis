# Latent-Bayesian-Linear-Regression-with-Coefficient-Table-and-Error-Diagnosis
Latent Bayesian Linear Regression with Coefficient Table and Error Diagnosis, Plots for relationship in Y &amp; X, K for #{latent class}, prior a&amp;l to parametrize


This comprehensive report provides a graduate-level decomposition of the framework, mathematical derivations, and output metrics of your **Multivariate Bayesian Latent Linear Regression** implementation.


# Part 0: How Others Download and Install It

1. Download the File from GitHub
Public GitHub repository URL: [https://github.com/cchalvin/Bayesianpac](https://github.com/cchalvin/Bayesianpac).

Click on the Bayesianpac_0.1.0.tar.gz file in the file list.

Click the Download button (or the Download raw file icon) to save the archive to their local machine.

2. Install the Archive via R
They can then install the package by pointing R directly to their local download path:

R
1. Install missing dependencies first if needed
dependencies <- c("mvtnorm", "ggplot2", "gridExtra")
new_packages <- dependencies[!(dependencies %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

2. Install your package from the downloaded .tar.gz file
install.packages(
  "path/to/downloads/Bayesianpac_0.1.0.tar.gz", 
  repos = NULL, 
  type = "source"
)

3. Verify the installation
library(Bayesianpac)






## 1. Architectural Framework & Application Domain

### Statistical Paradigm

The `BayesianLatentRegression` function implements a **Finite Mixture of Regularized Multivariate Linear Regressions** (often stylized as Latent Class Regression or Mixture-of-Experts with constant gating networks). It models a system where the relationship between a $p$-dimensional predictor vector $x_i$ and a $q$-dimensional response vector $y_i$ varies across $K$ unobserved sub-populations (latent classes).

Rather than executing a global multivariate regression—which forces a single average parameter matrix $B$ onto heterogeneous data—this framework clusters the conditional distributions $P(y_i \mid x_i)$. Concurrently, it regularizes the parameter space using an $L_2$ (Ridge) prior to guarantee numerical stability and mitigate overfitting under multi-collinearity or high-dimensional regimes.

### Data Preprocessing & Numerical Stabilization

The routine features structural safeguards engineered to maintain execution safety in high-throughput data pipelines:

* **Vectorized Conditional Imputation**: Missing values ($NA$) in the design matrix $X$ and response matrix $Y$ are dynamically mapped and imputed using column-wise expectations. For the response matrix, this occurs iteratively within the Expectation-Maximization (EM) loop to preserve conditional cluster profiles.
* **Tikhonov Regularization ($\Lambda_\epsilon$)**: To handle ill-posed inverse problems where the empirical Gram matrix is non-invertible, the parameter precision matrix is augmented:

$$\Lambda_\epsilon = \lambda_0 I_p + \epsilon I_p$$



This guarantees that the objective function is strictly convex and that the matrix inverse operations are well-conditioned.

### Application Scope

This architectural framework is ideal for modeling unobserved stratification in complex datasets, such as:

* **Econometrics / Quantitative Marketing**: Modeling distinct consumer segments whose purchasing choices respond differently to pricing and demographic variables.
* **Psychometrics & Educational Testing**: Evaluating performance profiles across distinct student latent cohorts where individual grouping data is withheld or missing.
* **Biostatistics**: Identifying latent phenotypic expressions or patient sub-types where treatment responses vary significantly due to hidden genetic covariates.

---

## 2. Mathematical Foundations & Algorithmic Derivations

### Generative Model Definition

Let $y_i \in \mathbb{R}^q$ be the row vector of responses for observation $i$, and $x_i \in \mathbb{R}^p$ be the corresponding row vector of predictors. The generative model assumes that observation $i$ belongs to an unobserved latent class $z_i \in \{1, \dots, K\}$ governed by a categorical distribution with mixing proportions $\pi = (\pi_1, \dots, \pi_K)^T$.

Conditional on membership in class $k$ ($z_i = k$), the response vector follows a multivariate Gaussian distribution:


$$y_i \mid x_i, z_i = k \sim \mathcal{N}_q(x_i B_k, \Sigma_k)$$

where $B_k \in \mathbb{R}^{p \times q}$ is the coefficient matrix for class $k$, and $\Sigma_k \in \mathbb{R}^{q \times q}$ is the class-specific symmetric positive-definite error covariance matrix.

### Prior Specifications

To incorporate the Bayesian regularized structure, we impose independent Conjugate/Objective priors on the parameter space $\Theta = \{\pi, \{B_k, \Sigma_k\}_{k=1}^K\}$:

1. **Dirichlet Prior on Mixing Proportions**:

$$P(\pi) \propto \prod_{k=1}^K \pi_k^{\alpha - 1}$$



where $\alpha \ge 1$ acts as a smoothing pseudo-count hyperparameter.
2. **Gaussian (Ridge) Prior on Regression Weights**:

$$P(B_k) \propto \exp\left( -\frac{1}{2} \text{tr}\left( B_k^T \Lambda_0 B_k \right) \right) = \exp\left( -\frac{\lambda_0}{2} \sum_{j=1}^q \beta_{k, \cdot, j}^T \beta_{k, \cdot, j} \right)$$



This structure shrinks the coefficients toward zero, indexed by the precision scalar $\lambda_0$.

### Expectation-Maximization (EM) MAP Derivation

The objective is to maximize the Maximum A Posteriori (MAP) energy functional. The complete-data log-posterior is formulated as:


$$\log P(\Theta \mid X, Y) = \sum_{i=1}^n \sum_{k=1}^K z_{ik} \left[ \log \pi_k + \log \mathcal{N}_q(y_i; x_i B_k, \Sigma_k) \right] + \log P(\pi) + \sum_{k=1}^K \log P(B_k) + \mathcal{C}$$

#### 1. E-Step (Expectation)

We compute the conditional expectation of the latent indicators $z_{ik}$ given the observed data and current parameter estimates $\Theta^{(t)}$. This defines the posterior responsibilities $\gamma_{ik}$:


$$\gamma_{ik} = \mathbb{E}[z_{ik} \mid x_i, y_i, \Theta^{(t)}] = \frac{\pi_k^{(t)} \mathcal{N}_q(y_i; x_i B_k^{(t)}, \Sigma_k^{(t)})}{\sum_{l=1}^K \pi_l^{(t)} \mathcal{N}_q(y_i; x_i B_l^{(t)}, \Sigma_l^{(t)})}$$

##### Numerical Underflow Prevention

To handle high-dimensional response vectors where joint normal densities collapse to absolute zero, the implementation processes operations in log-space using the **Log-Sum-Exp** identity:


$$\log \gamma_{ik} = \psi_{ik} - \max_l \psi_{il} - \log \sum_{l=1}^K \exp\left(\psi_{il} - \max_m \psi_{im}\right)$$


where $\psi_{ik} = \log \pi_k + \log \mathcal{N}_q(y_i; x_i B_k, \Sigma_k)$.

#### 2. M-Step (Maximization)

We maximize the expected complete-data log-posterior with respect to $\Theta$.

##### Update for Mixing Proportions $\pi_k$

Isolating terms containing $\pi_k$ and introducing the Lagrange multiplier $\mu$ for the constraint $\sum_k \pi_k = 1$:


$$\mathcal{L}(\pi) = \sum_{i=1}^n \sum_{k=1}^K \gamma_{ik} \log \pi_k + \sum_{k=1}^K (\alpha - 1)\log \pi_k - \mu \left(\sum_{k=1}^K \pi_k - 1\right)$$


Taking the partial derivative and setting it to zero:


$$\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{\sum_{i=1}^n \gamma_{ik} + \alpha - 1}{\pi_k} - \mu = 0 \implies \pi_k \propto N_k + \alpha - 1$$


Normalizing yields the optimal step:


$$\pi_k^{(t+1)} = \frac{N_k + \alpha - 1}{n + K(\alpha - 1)}, \quad \text{where } N_k = \sum_{i=1}^n \gamma_{ik}$$

##### Update for Regression Coefficients $B_k$

The objective component for the regularized class weight matrix is:


$$Q(B_k) = \sum_{i=1}^n \gamma_{ik} \left[ -\frac{1}{2} (y_i - x_i B_k) \Sigma_k^{-1} (y_i - x_i B_k)^T \right] - \frac{1}{2}\text{tr}(B_k^T \Lambda_\epsilon B_k)$$


Taking the matrix derivative with respect to $B_k$:


$$\frac{\partial Q}{\partial B_k} = X^T W_k (Y - X B_k) \Sigma_k^{-1} - \Lambda_\epsilon B_k = 0$$


where $W_k = \text{diag}(\gamma_{1k}, \dots, \gamma_{nk})$. Under coordinate-wise cross-products, this evaluates to the regularized normal equations:


$$B_k^{(t+1)} = \left( X^T W_k X + \Lambda_\epsilon \right)^{-1} X^T W_k Y$$

##### Update for Covariance Matrix $\Sigma_k$

Taking the derivative with respect to $\Sigma_k^{-1}$ and accounting for degrees of freedom adjustment:


$$\Sigma_k^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{ik} (y_i - x_i B_k)^T (y_i - x_i B_k) + \epsilon I_q}{N_k + q + 1}$$

### Asymptotic Matrix Covariances

For the Standard Baseline profile ($K=1$), the standard error matrix is mapped via the asymptotic variance-covariance matrix of the vectorization operator. By exploiting Kronecker structures under regularized assumptions:


$$\text{Cov}(\text{vec}(B_{\text{std}})) = \Sigma_{\text{std}} \otimes \left( X^T X + \Lambda_\epsilon \right)^{-1}$$

$$\text{SE}(B_{ij}) = \sqrt{\left[ \Sigma_{\text{std}} \right]_{jj} \cdot \left[ \left( X^T X + \Lambda_\epsilon \right)^{-1} \right]_{ii}}$$

### Localized Wald Bayesian Evidence Factor

To scale statistical significance tests into a Bayesian framework without nested sampling, the function evaluates a localized Wald-type evidence factor ($BF$). Under a normal asymptotic assumption for the posterior marginals:


$$z = \frac{\mu_{\text{posterior}}}{\sigma_{\text{posterior}}}$$


The localized Bayes Factor supporting the alternative hypothesis over the sharp null $H_0: \beta = 0$ is evaluated using the profile:


$$BF \approx \exp\left( \frac{1}{2} z^2 \right)$$

---

## 3. Interpretation of Analytical Outputs & Visualization Modality

The function returns a structured log and console stream containing three performance tables and a diagnostic visualization matrix.

### 1. Model Selection Matrix

The console displays an information-theoretic table comparing three nested architectures:

| Comparison | BIC_Diff | BayesFactor | Better_Model |
| --- | --- | --- | --- |
| Null vs Std | $\text{BIC}_{\text{null}} - \text{BIC}_{\text{std}}$ | $\text{BF}_{10}$ | Name of optimal path |
| Std vs Latent | $\text{BIC}_{\text{std}} - \text{BIC}_{\text{latent}}$ | $\text{BF}_{21}$ | Name of optimal path |

* **BIC Difference ($\Delta \text{BIC}$)**: A $\Delta \text{BIC} > 10$ provides strong evidence for the more complex model, indicating that the improvement in log-likelihood outweighs the parameter penalty.
* **Bayes Factor**: Quantifies the relative evidence provided by the data. For example, a `BayesFactor` of $150$ for `Std vs Latent` means the data is 150 times more likely under the Latent Class model than the single-class model.

### 2. Posterior Estimates & Regression Parameters Table

This table displays the parsed parameter coordinates across models and latent classes:

```text
Model         Class      Output      Variable         Mean      SD       Estimate_CI
Standard BLR  1 (1.00)   response1   predictor_cont   0.452     0.021    0.452 (0.411, 0.493) ***
Latent BLR    1 (0.48)   response1   predictor_cont   2.981     0.012    2.981 (2.957, 3.005) ***
Latent BLR    2 (0.52)   response1   predictor_cont  -1.954     0.015   -1.954 (-1.983, -1.925) ***

```

* **Class Notation**: The format `1 (0.48)` indicates Latent Class 1 with an estimated mixing proportion $\hat{\pi}_1 = 0.48$.
* **Evidence Coding**: The string suffixes (`***`, `**`, `*`, `.`) code the magnitude of the localized Bayes Factor, pinpointing which variables drive class separation.

### 3. Covariate Contraction Summary

This diagnostic table measures information gain by comparing the prior variance against the localized posterior variance:

$$\text{SD Ratio} = \frac{\sigma_{\text{posterior}}}{\sigma_{\text{prior}}}$$

* **Interpretation**: An `SD_Ratio` near $1.0$ indicates that the covariate provides little to no new information. Conversely, an `SD_Ratio` approaching $0.0$ signifies substantial **posterior contraction**, meaning the data has significantly reduced parameter uncertainty.
* **Mean Change %**: Tracks parameter relocation from the prior initialization baseline to the final posterior mode:

$$\text{Mean Change \%} = \frac{\mu_{\text{post}} - \mu_{\text{prior}}}{\mu_{\text{post}}} \times 100\%$$




* **Continuous Features**: Displays a scatter plot of the data points, colored by their highest posterior probability class assignment ($\arg\max_k \gamma_{ik}$). The solid lines represent the class-specific regression lines, surrounded by a $95\%$ shaded confidence ribbon ($x_i \hat{B}_k \pm 1.96 \hat{\sigma}_{kd}$). The **dashed black line** shows the global OLS fit, illustrating how a single line misrepresents the two distinct underlying trends.
* **Categorical Features**: Displays grouped, side-by-side boxplots for each factor level, split by latent class. This visualizes differences in both the conditional mean and variance across discrete groups, making it easy to see how class assignments vary across categorical levels.













