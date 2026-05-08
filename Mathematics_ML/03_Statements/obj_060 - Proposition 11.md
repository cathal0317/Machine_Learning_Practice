---
id: obj_060
title: "Proposition 11"
types:
  - proposition
page_start: 27
page_end: 27
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_062
  - obj_063
  - obj_064
  - obj_065
  - obj_066
  - obj_067
prerequisites:
  - obj_038
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 11

## Conceptual overview
If you have $d$ sub-Gaussian random variables, the expected value of the largest one is bounded by $\sigma \sqrt{2 \log d}$. This holds even if the variables are not independent.

## Why it matters
This is a very general and powerful result in probability that limits how fast the 'worst-case' outcome grows as the number of possible outcomes increases.

## Active recall
> [!question]- What is the mathematical bound provided by Proposition 11?
> $\mathbb{E}(\max_j W_j) \le \sigma \sqrt{2 \log d}$.

## Exact statement
Suppose $W_1, \dots, W_d$ are mean zero and sub-Gaussian with parameter $\sigma > 0$ (not necessarily independent). Then, 
$$
\mathbb{E}(\max_j W_j) \le \sigma \sqrt{2 \log d}.
$$

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_038 - Proposition 3]]
