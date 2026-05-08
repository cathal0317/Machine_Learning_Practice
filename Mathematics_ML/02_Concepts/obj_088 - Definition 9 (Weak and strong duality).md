---
id: obj_088
title: "Definition 9 (Weak and strong duality)"
types:
  - definition
page_start: 37
page_end: 37
parent_id: "obj_086"
children_ids: []
sibling_ids:
  - obj_084
  - obj_087
  - obj_089
  - obj_090
  - obj_092
  - obj_094
  - obj_095
  - obj_096
prerequisites:
  - obj_087
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 9 (Weak and strong duality)

## Conceptual overview
Weak duality states that the maximum of the dual function is always a lower bound for the minimum of the primal objective. Strong duality occurs when this lower bound is tight, meaning the primal and dual optimal values are equal, which is common in convex optimization.

## Why it matters
Strong duality is the bridge that allows one to solve the dual problem and guarantee that the result is the same as solving the primal problem directly.

## Active recall
> [!question]- What is the mathematical condition for strong duality?
> Strong duality holds if $\sup_{\xi \in \mathbb{R}^n, \nu \in \mathbb{R}^m_{\ge 0}} D(\xi, \nu) = \inf_{w \in \mathcal{F}} f_0(w)$.

## Mental picture
Imagine two people measuring the same distance from opposite ends: weak duality means they haven't met yet, while strong duality means they are touching hands at the exact same point.

## Common confusions
Strong duality is not a universal property; it requires specific conditions (like convexity and Slater's condition) to hold, unlike weak duality which is always true.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Prerequisites:**
- [[obj_087 - Definition 8 (The dual problem)]]
