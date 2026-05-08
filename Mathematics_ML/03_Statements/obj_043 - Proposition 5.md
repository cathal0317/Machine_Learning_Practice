---
id: obj_043
title: "Proposition 5"
types:
  - proposition
page_start: 22
page_end: 22
parent_id: "obj_035"
children_ids:
  - obj_042
sibling_ids:
  - obj_032
  - obj_033
  - obj_034
  - obj_036
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_044
prerequisites:
  - obj_036
used_in:
  - obj_044
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 5

## Conceptual overview
If multiple independent random variables are each sub-Gaussian, their linear combination is also sub-Gaussian. The resulting parameter is the square root of the sum of the squared individual parameters, weighted by the linear coefficients.

## Why it matters
This result allows us to extend tail bounds from single variables to empirical averages (which are sums of variables), enabling the derivation of Hoeffding's inequality.

## Exact statement
Suppose $W_1, \dots, W_n$ are independent and each $W_i$ is sub-Gaussian with parameter $\sigma_i$. Then, for any $\gamma \in \mathbb{R}^n$, $\sum_{i=1}^n \gamma_i W_i$ is sub-Gaussian with parameter $(\sum_i \gamma_i^2 \sigma_i^2)^{1/2}$.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Children:**
- [[obj_042 - Proof of Proposition 5]]

**Prerequisites:**
- [[obj_036 - Definition 2]]

**Used in:**
- [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]
