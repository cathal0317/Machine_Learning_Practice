---
id: obj_092
title: "Example 16"
types:
  - example
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
  - obj_094
  - obj_095
  - obj_096
prerequisites:
  - obj_088
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 16

## Conceptual overview
This example calculates the dual of $\min \|x\|^2$ subject to $Ax = b$. It derives the dual objective $-1/4 \|A^\top z\|^2 + \langle b, z \rangle$ by explicitly minimizing the Lagrangian and shows that strong duality holds when a feasible point exists.

## Why it matters
It provides a concrete, solvable case where strong duality applies, helping students understand the steps of dual derivation.

## Active recall
> [!question]- What is the dual problem for the least squares problem $\min \|x\|^2$ s.t. $Ax = b$?
> The dual problem is $\max_z -1/4 \|A^\top z\|^2 + \langle b, z \rangle$.

## When to use
Use this example to practice deriving dual functions for quadratic objectives with equality constraints.

## Core pattern
Form the Lagrangian, set its gradient with respect to the primal variable to zero to find the infimum, and substitute back to get the dual function.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Prerequisites:**
- [[obj_088 - Definition 9 (Weak and strong duality)]]
