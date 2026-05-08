---
id: obj_179
title: "Lemma 27 (KL divergence between Gaussian distributions)"
types:
  - lemma
page_start: 76
page_end: 76
parent_id: "obj_181"
children_ids:
  - obj_182
sibling_ids:
  - obj_183
prerequisites: []
used_in:
  - obj_181
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Lemma 27 (KL divergence between Gaussian distributions)

## Conceptual overview
This lemma provides the analytic closed-form solution for the KL divergence between two multivariate Gaussian distributions with diagonal covariance matrices, which is essential for VAE implementation.

## Why it matters
It allows the regularization part of the VAE objective to be computed exactly without needing stochastic approximations, reducing the variance of gradients.

## Exact statement
Let $q = \mathcal{N}(\mu, \sigma^2 Id)$ and $p = \mathcal{N}(\nu, \eta^2 Id)$ be Gaussian distributions on $\mathbb{R}^d$. Then, 
$$
KL(q || p) = d \log(\eta/\sigma) + \frac{1}{2\eta^2} ||\mu - \nu||^2 + \frac{d}{2} (\frac{\sigma^2}{\eta^2} - 1)
$$

## Links
**Parent:** [[obj_181 - Computation of gradients]]

**Children:**
- [[obj_182 - Proof of Lemma 27]]

**Used in:**
- [[obj_181 - Computation of gradients]]
