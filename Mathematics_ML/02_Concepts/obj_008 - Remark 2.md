---
id: obj_008
title: "Remark 2"
types:
  - remark
page_start: 6
page_end: 6
parent_id: "obj_005"
children_ids: []
sibling_ids:
  - obj_003
  - obj_004
  - obj_007
  - obj_009
prerequisites:
  - obj_009
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 2

## Conceptual overview
This remark identifies the theoretical minimizer of the squared error risk in regression. For a quantitative response $Y$, the function $h$ that minimizes $R(h) = E(h(X) - Y)^2$ is given by the conditional expectation $h(x) = E(Y | X = x)$. This function is formally termed the 'regression function'.

## Why it matters
It establishes that if we knew the full joint distribution, the best possible prediction for regression is simply the conditional mean.

## Active recall
> [!question]- What function minimizes the squared error risk in regression?
> The risk is minimized by the conditional expectation $h(x) = \mathbb{E}(Y \mid X = x)$.

## Mental picture
In a scatter plot, the regression function is the 'average' line that passes through the center of the vertical distribution of data points for any given $x$.

## Common confusions
Assuming that the conditional expectation is easy to compute in practice (it requires full knowledge of $P_0$).

## Links
**Parent:** [[obj_005 - 1.1 Classification and regression]]

**Prerequisites:**
- [[obj_009 - The Risk]]
