---
id: obj_042
title: "Proof of Proposition 5"
types:
  - proof
page_start: 22
page_end: 22
parent_id: "obj_043"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_043
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 5

## Conceptual overview
This proof uses the property that the expectation of a product of independent random variables is the product of their expectations. By factorizing the joint moment generating function, the result follows from the sub-Gaussian definition of the individual terms.

## Why it matters
Independence allows sub-Gaussian parameters to behave predictably under summation, similar to how variances add for Gaussian variables.

## Active recall
> [!question]- What property of independent variables is central to the proof of Proposition 5?
> The fact that for independent variables $X_i$, the expectation of the product $\mathbb{E}(\prod e^{\alpha \gamma_i X_i})$ is the product of the expectations $\prod \mathbb{E}(e^{\alpha \gamma_i X_i})$.

## Proves
[[obj_043 - Proposition 5]]

## Proof skeleton
**Goal.** Show the sum of independent sub-Gaussians is sub-Gaussian.

**Strategy.** Factorize the MGF of the sum into a product of individual MGFs.

- Write the MGF of the sum $\mathbb{E}(e^{\alpha \sum \gamma_i W_i})$ as $\prod \mathbb{E}(e^{\alpha \gamma_i W_i})$ by independence.

- Apply the sub-Gaussian bound $e^{\alpha^2 \gamma_i^2 \sigma_i^2 / 2}$ to each term in the product.

- Re-exponentiate the product of exponentials to find the resulting parameter $(\sum \gamma_i^2 \sigma_i^2)^{1/2}$.

**Conclusion.** The resulting expression matches the sub-Gaussian definition with the combined parameter.

## Full proof with commentary
- We assume each $W_i$ has mean zero and write the moment generating function for the linear combination $\sum \gamma_i W_i$.
   - *Why:* By the properties of exponents, $\mathbb{E}(e^{\alpha \sum \gamma_i W_i}) = \mathbb{E}(\prod_{i=1}^n e^{\alpha \gamma_i W_i})$.
 - Utilize the independence of the random variables $W_i$ to factorize the expectation of the product.
   - *Why:* For independent random variables, the expectation of a product of functions of each variable equals the product of their individual expectations: $\prod_{i=1}^n \mathbb{E}(e^{\alpha \gamma_i W_i})$.
 - Apply the definition of sub-Gaussianity for each individual variable $W_i$.
   - *Why:* Since each $W_i$ is sub-Gaussian with parameter $\sigma_i$, we have $\mathbb{E}(e^{(\alpha \gamma_i) W_i}) \le e^{(\alpha \gamma_i)^2 \sigma_i^2 / 2}$.
 - Multiply these individual bounds back together.
   - *Why:* Summing the exponents in the product, we get $\exp( \alpha^2 \sum_{i=1}^n \gamma_i^2 \sigma_i^2 / 2 )$.
 **Conclusion.** The resulting expression matches the definition of a sub-Gaussian variable with parameter $(\sum \gamma_i^2 \sigma_i^2)^{1/2}$.

## Links
**Parent:** [[obj_043 - Proposition 5]]

**Prerequisites:**
- [[obj_043 - Proposition 5]]
