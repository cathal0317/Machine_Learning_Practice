---
id: obj_080
title: "Proposition 15 (First order optimality condition - unconstrained)"
types:
  - proposition
page_start: 35
page_end: 35
parent_id: "obj_071"
children_ids:
  - obj_078
sibling_ids:
  - obj_072
  - obj_074
  - obj_076
  - obj_077
  - obj_081
  - obj_082
prerequisites:
  - obj_076
used_in:
  - obj_082
  - obj_111
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 15 (First order optimality condition - unconstrained)

## Conceptual overview
For a differentiable convex function on an unconstrained domain, a point is a global minimizer if and only if its gradient is zero. This effectively transforms the problem of global optimization into the problem of solving a system of equations.

## Why it matters
This is the theoretical foundation for practically all derivative-based optimization methods in machine learning.

## Active recall
> [!question]- What is the necessary and sufficient condition for optimality in unconstrained convex optimization?
> $\nabla f(w^*) = 0$.

## Exact statement
Suppose that $f$ is convex and differentiable. Then, $w^* \in \text{argmin}(f)$ if and only if $\nabla f(w^*) = 0$.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Children:**
- [[obj_078 - Proof of Proposition 15]]

**Prerequisites:**
- [[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]

**Used in:**
- [[obj_082 - Remark 10]]
- [[obj_111 - Example 17 (Least squares)]]
