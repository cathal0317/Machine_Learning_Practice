---
id: obj_050
title: "Remark 7"
types:
  - remark
page_start: 24
page_end: 24
parent_id: "obj_048"
children_ids: []
sibling_ids:
  - obj_047
  - obj_049
prerequisites:
  - obj_047
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 7

## Conceptual overview
This remark notes that the right-hand side of the deviation inequality can be computed from data (unlike the true risk). It suggests this can be used to select the best model complexity, a process called structural risk minimization.

## Why it matters
It explains how these theoretical tail bounds are actually used in practice to prevent overfitting by penalizing large $\mathcal{H}$.

## Active recall
> [!question]- What is the main idea behind structural risk minimization?
> Choosing a hypothesis class $\mathcal{H}$ that minimizes the sum of the empirical risk and the estimation error bound.

## Mental picture
Imagine a set of nested circles representing increasingly complex models. Structural risk minimization is the search for the circle that best balances fit (inner regions) with reliability (outer boundaries).

## Common confusions
Students often think the bound $R(h)$ is the actual value, when it is actually an upper limit that holds with high probability.

## Links
**Parent:** [[obj_048 - 1.6.2 Finite hypothesis classes]]

**Prerequisites:**
- [[obj_047 - Theorem 7]]
