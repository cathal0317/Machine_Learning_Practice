---
id: obj_090
title: "Lemma 17"
types:
  - lemma
page_start: 37
page_end: 37
parent_id: "obj_086"
children_ids:
  - obj_091
sibling_ids:
  - obj_084
  - obj_087
  - obj_088
  - obj_089
  - obj_092
  - obj_094
  - obj_095
  - obj_096
prerequisites:
  - obj_087
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Lemma 17

## Conceptual overview
This lemma proves two fundamental properties of the dual function $D$: it is concave, regardless of whether the primal problem is convex, and it satisfies the weak duality inequality $D(\xi, \nu) \le f_0(w)$ for all feasible $w$.

## Why it matters
These properties ensure that solving the dual problem is always a convex optimization problem (maximizing a concave function) and that any dual feasible solution provides a guaranteed lower bound on the primal cost.

## Active recall
> [!question]- Is the dual function always concave?
> Yes, the dual function is always concave because it is the pointwise infimum of affine functions of the multipliers, even if the primal objective is not convex.

## Exact statement
The dual function $D$ is concave. Moreover, for all $\xi \in \mathbb{R}^n$ and $\nu \in \mathbb{R}^m_{\ge 0}$, $D(\xi, \nu) \le \inf_{w \in \mathcal{F}} f_0(w)$.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Children:**
- [[obj_091 - Proof of Lemma 17]]

**Prerequisites:**
- [[obj_087 - Definition 8 (The dual problem)]]
