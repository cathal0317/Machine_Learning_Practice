---
id: obj_087
title: "Definition 8 (The dual problem)"
types:
  - definition
page_start: 37
page_end: 37
parent_id: "obj_086"
children_ids: []
sibling_ids:
  - obj_084
  - obj_088
  - obj_089
  - obj_090
  - obj_092
  - obj_094
  - obj_095
  - obj_096
prerequisites:
  - obj_084
used_in:
  - obj_088
  - obj_101
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 8 (The dual problem)

## Conceptual overview
The dual problem is the task of maximizing the Lagrange dual function $D(\xi, \nu)$ over the Lagrange multipliers. Unlike the primal problem, which may be complex, the dual problem is always a concave maximization problem, making it potentially easier to solve or analyze.

## Why it matters
Defining the dual problem is the first step in applying duality theory to find bounds or solutions for constrained optimization problems.

## Active recall
> [!question]- How is the dual optimization problem mathematically formulated?
> It is formulated as $\sup_{\xi \in \mathbb{R}^n, \nu \in \mathbb{R}^m} D(\xi, \nu)$ subject to the constraint $\nu \ge 0$.

## Mental picture
Think of the dual problem as trying to find the 'best possible' lower bound for the primal minimization by adjusting the 'penalties' associated with each constraint.

## Common confusions
One might incorrectly assume the dual problem's constraints apply to all multipliers, but only the multipliers $\nu$ associated with inequality constraints must be non-negative.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Prerequisites:**
- [[obj_084 - Definition 7 (Duality)]]

**Used in:**
- [[obj_088 - Definition 9 (Weak and strong duality)]]
- [[obj_101 - 2.3.3 The dual problem for SVM]]
