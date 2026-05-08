---
id: obj_009
title: "The Risk"
types:
  - definition
page_start: 6
page_end: 6
parent_id: "obj_005"
children_ids: []
sibling_ids:
  - obj_003
  - obj_004
  - obj_007
  - obj_008
prerequisites:
  - obj_004
used_in:
  - obj_011
  - obj_018
  - obj_025
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# The Risk

## Conceptual overview
Risk $R(h)$ is the central objective of statistical learning theory. It is defined as the expectation of the loss function $\ell(h(x), y)$ over the unknown joint probability distribution $P_0$ of features $X$ and labels $Y$. Because $P_0$ is typically unknown, we cannot calculate $R(h)$ directly, necessitating the use of empirical risk proxies.

## Why it matters
Risk represents the true generalization error of a model on unseen data from the same population.

## Active recall
> [!question]- State the formal integral definition of Risk $R(h)$.
> $$
> R(h) := \int_{\mathcal{X} \times \mathcal{Y}} \ell(h(x), y) dP_0(x, y) = \mathbb{E}(\ell(h(X), Y))
> $$

## Mental picture
Imagine you are throwing darts (predictions) at a moving target (reality); risk is the average score you would get if you threw infinitely many darts.

## Common confusions
Confusing the integral over the joint distribution with an average over a finite sample.

## Links
**Parent:** [[obj_005 - 1.1 Classification and regression]]

**Prerequisites:**
- [[obj_004 - The loss function]]

**Used in:**
- [[obj_011 - 1.2 The Bayes classifier]]
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
- [[obj_025 - Definition 1]]
