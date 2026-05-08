---
id: obj_095
title: "Theorem 18 (Slater's constraint quantification)"
types:
  - theorem
page_start: 38
page_end: 38
parent_id: "obj_086"
children_ids:
  - obj_093
sibling_ids:
  - obj_084
  - obj_087
  - obj_088
  - obj_089
  - obj_090
  - obj_092
  - obj_094
  - obj_096
prerequisites: []
used_in:
  - obj_096
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 18 (Slater's constraint quantification)

## Conceptual overview
Slater's condition provides a sufficient condition for strong duality in convex optimization. It requires that the problem be convex and that there exists a 'strictly feasible' point, i.e., a point that satisfies all linear constraints and strictly satisfies all inequality constraints.

## Why it matters
It is the standard tool used to check if solving the dual problem will yield the true primal minimum for a convex program.

## Active recall
> [!question]- What are the requirements for Slater's condition?
> The objective and inequality constraints must be convex, and there must exist a point $w$ such that $Aw=b$ and $f_i(w) < 0$ for all inequality constraints.

## Exact statement
Assume that $f_i$ are convex for $i = 0, \dots, m$ with $\text{dom}(f_0) = \mathbb{R}^p$. If there exists $w \in \mathbb{R}^p$ such that $Aw = b$ and $f_\ell(w) < 0$ for all $\ell = 1, \dots, m$, then strong duality holds for the problem.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Children:**
- [[obj_093 - Proof of Theorem 18]]

**Used in:**
- [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]
