---
id: obj_094
title: "The Karush-Kuhn-Tucker (KKT) conditions discussion"
types:
  - discussion
page_start: 38
page_end: 38
parent_id: "obj_086"
children_ids: []
sibling_ids:
  - obj_084
  - obj_087
  - obj_088
  - obj_089
  - obj_090
  - obj_092
  - obj_095
  - obj_096
prerequisites: []
used_in:
  - obj_096
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# The Karush-Kuhn-Tucker (KKT) conditions discussion

## Conceptual overview
This discussion derives the relationship between optimal primal variables and dual multipliers under strong duality. It introduces the core concepts of stationarity and complementary slackness by showing that inequalities must hold as equalities at the optimum.

## Why it matters
It bridge the gap between abstract duality theory and practical conditions used to solve or verify optimization problems.

## Active recall
> [!question]- What is the complementary slackness condition?
> It is the condition that $\nu^*_i f_i(w^*) = 0$, meaning that either the $i$-th constraint is active ($f_i(w^*) = 0$) or the corresponding multiplier is zero ($\nu^*_i = 0$).

## Narrative flow
The discussion starts with the assumption of strong duality and primal-dual optimality. It uses algebraic equalities to deduce that individual terms in the Lagrangian must be zero, leading directly to complementary slackness and stationarity.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Used in:**
- [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]
