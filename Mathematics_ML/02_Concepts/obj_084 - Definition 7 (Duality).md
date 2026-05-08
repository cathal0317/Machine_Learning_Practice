---
id: obj_084
title: "Definition 7 (Duality)"
types:
  - definition
page_start: 36
page_end: 37
parent_id: "obj_086"
children_ids: []
sibling_ids:
  - obj_087
  - obj_088
  - obj_089
  - obj_090
  - obj_092
  - obj_094
  - obj_095
  - obj_096
prerequisites: []
used_in:
  - obj_087
  - obj_101
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 7 (Duality)

## Conceptual overview
The Lagrange function combines the objective and constraints into a single expression using multipliers $\xi$ (for equality) and $\nu$ (for inequality). The Lagrange dual function is then defined as the infimum of this function over the primal variables, providing a lower bound on the primal optimal value.

## Why it matters
Duality allows us to solve optimization problems in a different 'space' (the dual space), which is often easier or provides better theoretical insights.

## Active recall
> [!question]- What are the Lagrange multipliers in the Lagrange function?
> They are vectors $\xi$ and $\nu$ that weight the constraints $Aw - b$ and $f_k(w)$ in the combined objective.

> [!question]- How is the Lagrange dual function $D(\xi, \nu)$ calculated?
> $D(\xi, \nu) = \inf_{w \in \mathbb{R}^p} L(w, \xi, \nu)$.

## Mental picture
Imagine the constraints as 'punishments' whose severity is determined by the multipliers. The dual function finds the best-case scenario (minimum) for a given set of punishment levels.

## Common confusions
Forgetting that the multipliers $\nu$ for inequality constraints must be non-negative.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Used in:**
- [[obj_087 - Definition 8 (The dual problem)]]
- [[obj_101 - 2.3.3 The dual problem for SVM]]
