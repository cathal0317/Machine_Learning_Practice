---
id: obj_083
title: "Example 14"
types:
  - example
page_start: 36
page_end: 36
parent_id: "obj_085"
children_ids: []
sibling_ids:
  - obj_086
prerequisites:
  - obj_018
  - obj_085
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 14

## Conceptual overview
This example shows how a typical machine learning task (regularized regression or constrained ERM) can be translated into the standard mathematical form $\min f_0(w)$ subject to $f_i(w) \leq 0$. Here, the constraint $\|w\| \leq K$ becomes $f_1(w) = \sum w_i^2 - K^2 \leq 0$.

## Why it matters
It bridges the gap between learning theory (hypothesis classes) and optimization algorithms.

## Active recall
> [!question]- How is a norm constraint $\|w\| \leq K$ expressed in standard optimization form?
> It is expressed as an inequality constraint $f_1(w) = \sum_{i} w_i^2 - K^2 \leq 0$.

## When to use
When converting a high-level learning problem into a formal optimization problem solvable by standard solvers.

## Core pattern
Identifying the objective function and the inequality constraints to match the standard form $(P)$.

## Links
**Parent:** [[obj_085 - 2.2 Constrained optimization]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
- [[obj_085 - 2.2 Constrained optimization]]
