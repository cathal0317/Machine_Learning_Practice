---
id: obj_025
title: "Definition 1"
types:
  - definition
page_start: 16
page_end: 16
parent_id: "obj_028"
children_ids: []
sibling_ids:
  - obj_026
  - obj_027
  - obj_029
prerequisites:
  - obj_011
used_in:
  - obj_029
  - obj_056
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 1

## Conceptual overview
Excess risk $\mathcal{E} = R(\hat{h}) - R(h^*)$ measures how much more risk an estimator $\hat{h}$ incurs compared to the theoretical optimal Bayes classifier $h^*$. It quantifies the gap between a learned model and the best possible performance achievable for a given distribution.

## Why it matters
It is the central quantity in statistical learning theory used to bound the performance of learning algorithms.

## Active recall
> [!question]- Write the formula for excess risk $\mathcal{E}$.
> $\mathcal{E} := R(\hat{h}) - R(h^*)$.

## Mental picture
Imagine a leaderboard of classifiers: the Bayes classifier is the undisputed champion at the top, and the excess risk is the distance you are trailing behind them.

## Common confusions
Confusing excess risk with empirical risk; excess risk compares to the optimal Bayes risk, while empirical risk is calculated on training data labels.

## Links
**Parent:** [[obj_028 - 1.5 Excess risk]]

**Prerequisites:**
- [[obj_011 - 1.2 The Bayes classifier]]

**Used in:**
- [[obj_029 - 1.5.1 Decomposition of excess risk]]
- [[obj_056 - Theorem 9]]
