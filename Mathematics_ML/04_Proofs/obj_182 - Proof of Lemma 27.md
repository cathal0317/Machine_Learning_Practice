---
id: obj_182
title: "Proof of Lemma 27"
types:
  - proof
page_start: 77
page_end: 77
parent_id: "obj_179"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_179
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Lemma 27

## Conceptual overview
The proof uses the properties of exponential families and independence to decompose the multivariate KL divergence into a sum of univariate ones, which are then solved by direct integration.

## Why it matters
Confirms the analytic formula used for the VAE regularization term, ensuring mathematical correctness of the implementation.

## Active recall
> [!question]- What property of KL divergence allows simplifying the multivariate Gaussian case?
> The additivity of KL divergence under product measures, given that coordinates are independent.

## Proves
[[obj_179 - Lemma 27 (KL divergence between Gaussian distributions)]]

## Proof skeleton
**Goal.** Derive the formula for $KL(q || p)$ for Gaussians.

**Strategy.** Decompose into dimensions and use the log-ratio of densities.

- Factorize $q$ and $p$ into products $\prod q_i$ and $\prod p_i$.

- Write $KL(q || p) = \sum KL(q_i || p_i)$.

- Compute the 1D log-ratio: $\log(q(t)/p(t)) = \log(\eta/\sigma) - \frac{(t-\mu)^2}{2\sigma^2} + \frac{(t-\nu)^2}{2\eta^2}$.

- Take the expectation under $q$, noting $\mathbb{E}_q((t-\mu)^2) = \sigma^2$ and $\mathbb{E}_q((t-\nu)^2) = \sigma^2 + (\mu-\nu)^2$.

**Conclusion.** Substitute these into the sum to reach the final formula.

## Full proof with commentary
- For multivariate Gaussian distributions with diagonal covariance matrices, we decompose the global KL divergence into a sum of univariate components.
   - *Why:* Because the coordinates are independent, the joint densities factorize as $q(x) = \prod q_i(x_i)$ and $p(x) = \prod p_i(x_i)$. The additivity of KL divergence under product measures implies $KL(q || p) = \sum_{i=1}^d KL(q_i || p_i)$.
 - We now calculate the KL divergence for a single dimension with $q \sim \mathcal{N}(\mu, \sigma^2)$ and $p \sim \mathcal{N}(\nu, \eta^2)$.
 - Form the log-ratio of the probability densities.
   - *Why:* 
$$
\log(q(t)/p(t)) = \log\left( \frac{\eta}{\sigma} \right) - \frac{(t-\mu)^2}{2\sigma^2} + \frac{(t-\nu)^2}{2\eta^2}.
$$

 - Take the expectation of this log-ratio with respect to $q$.
   - *Why:* The expectation of the second term is $\mathbb{E}_q(t-\mu)^2 / (2\sigma^2) = \sigma^2 / (2\sigma^2) = 1/2$. For the third term, we use the variance identity $\mathbb{E}_q(t-\nu)^2 = \mathbb{E}_q(t-\mu+\mu-\nu)^2 = \sigma^2 + (\mu-\nu)^2$.
 - Combine the expectations into the univariate formula.
   - *Why:* 
$$
KL(q || p) = \log(\eta/\sigma) - 1/2 + \frac{\sigma^2 + (\mu-\nu)^2}{2\eta^2} = \log(\eta/\sigma) + \frac{(\mu-\nu)^2}{2\eta^2} + \frac{1}{2}(\sigma^2/\eta^2 - 1).
$$

 **Conclusion.** Summing this result over all $d$ dimensions (where parameters are identical across dimensions in the statement) yields the multivariate formula given in Lemma 27.

## Links
**Parent:** [[obj_179 - Lemma 27 (KL divergence between Gaussian distributions)]]

**Prerequisites:**
- [[obj_179 - Lemma 27 (KL divergence between Gaussian distributions)]]
