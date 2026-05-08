---
id: obj_077
title: "Example 13 (Least squares)"
types:
  - example
page_start: 35
page_end: 35
parent_id: "obj_071"
children_ids: []
sibling_ids:
  - obj_072
  - obj_074
  - obj_076
  - obj_080
  - obj_081
  - obj_082
prerequisites:
  - obj_080
used_in:
  - obj_111
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 13 (Least squares)

## Conceptual overview
For the objective $f(w) = \|Aw - b\|^2$, the gradient is $2A^\top(Aw - b)$. Setting this to zero yields the normal equations, providing the optimal weights for linear regression.

## Why it matters
It shows that the standard closed-form solution for linear regression is a direct consequence of convex optimality theory.

## Active recall
> [!question]- What are the normal equations for least squares?
> $A^\top A w^* = A^\top b$.

## When to use
When solving linear regression or any unconstrained quadratic minimization problem.

## Core pattern
Calculating the gradient, setting it to zero, and solving the resulting linear system.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Prerequisites:**
- [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]

**Used in:**
- [[obj_111 - Example 17 (Least squares)]]
