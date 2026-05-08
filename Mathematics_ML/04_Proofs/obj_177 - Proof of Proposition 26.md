---
id: obj_177
title: "Proof of Proposition 26"
types:
  - proof
page_start: 75
page_end: 76
parent_id: "obj_175"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_175
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 26

## Conceptual overview
The proof demonstrates how to manipulate the integral for marginal likelihood into an expectation over a chosen distribution $q$, allowing for the application of Jensen's inequality.

## Why it matters
It provides the foundational verification for the objective function used in VAE training.

## Active recall
> [!question]- Which property of logarithms is utilized in the proof of Proposition 26?
> The concavity of the logarithm function is used to apply Jensen's inequality.

## Proves
[[obj_175 - Proposition 26 (ELBO)]]

## Proof skeleton
**Goal.** Prove $\log(p_\theta(x)) \geqslant \mathbb{E}_{z \sim q}(\log(p_\theta(x|z))) - KL(q || p_Z)$.

**Strategy.** Rewrite the marginal likelihood using an expectation over $q$ and apply Jensen's inequality.

- Write $\log(p_\theta(x)) = \log \int p_\theta(x, z) dz = \log \int \frac{p_\theta(x, z)}{q(z)} q(z) dz$.

- Recognize the integral as an expectation: $\log \mathbb{E}_{z \sim q}(\frac{p_\theta(x, z)}{q(z)})$.

- Apply Jensen's inequality: $\log \mathbb{E}(\dots) \geqslant \mathbb{E} \log(\dots)$.

- Substitute $p_\theta(x, z) = p_\theta(x|z) p_Z(z)$ into the expression.

**Conclusion.** Separate the log terms to get the reconstruction term and the negative KL divergence term.

## Full proof with commentary
- We begin by expressing the log-marginal likelihood of the data as a log-integral over the latent space.
   - *Why:* $\log p_\theta(x) = \log \int p_\theta(x, z) dz$.
 - Introduce an arbitrary distribution $q(z)$ into the integral to reframe it as an expectation.
   - *Why:* By multiplying and dividing the integrand by $q(z)$, we write 
$$
\log p_\theta(x) = \log \int \frac{p_\theta(x, z)}{q(z)} q(z) dz = \log \mathbb{E}_{z \sim q}\left( \frac{p_\theta(x, z)}{q(z)} \right).
$$

 - Apply Jensen's inequality to move the logarithm inside the expectation.
   - *Why:* Since the logarithm is a concave function, $\log \mathbb{E}(f(Z)) \ge \mathbb{E}(\log f(Z))$. This yields the lower bound 
$$
\log p_\theta(x) \ge \mathbb{E}_{z \sim q}\left( \log \frac{p_\theta(x, z)}{q(z)} \right).
$$

 - Expand the joint density in the numerator using Bayes' rule.
   - *Why:* Substituting $p_\theta(x, z) = p_\theta(x|z) p_Z(z)$ and applying log properties, the expression becomes 
$$
\mathbb{E}_{z \sim q}(\log p_\theta(x|z) + \log p_Z(z) - \log q(z)) = \mathbb{E}_{z \sim q}(\log p_\theta(x|z)) - \mathbb{E}_{z \sim q}\left(\log \frac{q(z)}{p_Z(z)}\right).
$$

 **Conclusion.** The final terms represent the expected reconstruction log-likelihood and the negative KL divergence between the approximate posterior and the prior, respectively.

## Links
**Parent:** [[obj_175 - Proposition 26 (ELBO)]]

**Prerequisites:**
- [[obj_175 - Proposition 26 (ELBO)]]
