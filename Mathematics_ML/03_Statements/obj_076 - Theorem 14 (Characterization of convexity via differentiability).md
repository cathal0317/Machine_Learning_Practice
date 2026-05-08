---
id: obj_076
title: "Theorem 14 (Characterization of convexity via differentiability)"
types:
  - theorem
page_start: 34
page_end: 34
parent_id: "obj_071"
children_ids:
  - obj_075
sibling_ids:
  - obj_072
  - obj_074
  - obj_077
  - obj_080
  - obj_081
  - obj_082
prerequisites:
  - obj_072
used_in:
  - obj_080
  - obj_117
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 14 (Characterization of convexity via differentiability)

## Conceptual overview
This theorem provides two practical tests: a first-order test (the function graph lies above its tangents) and a second-order test (the Hessian matrix is positive semi-definite everywhere). If the Hessian is strictly positive definite, the function is strictly convex.

## Why it matters
These criteria allow us to check if a machine learning loss function is convex simply by calculating its derivatives.

## Active recall
> [!question]- What is the second-order characterization of convexity?
> A twice-differentiable function is convex if and only if its Hessian $\nabla^2 f(v)$ is positive semi-definite for all $v$.

## Exact statement
i) If $f$ is differentiable, then $f$ is convex iff for all $v, w \in \mathbb{R}^p, f(w) \geq f(v) + \nabla f(v)^\top (w-v)$. ii) If $f$ is twice-differentiable, then $f$ is convex iff $\nabla^2 f(v)$ is positive semi-definite for all $v$.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Children:**
- [[obj_075 - Proof of Theorem 14]]

**Prerequisites:**
- [[obj_072 - Definition 6]]

**Used in:**
- [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]
- [[obj_117 - Proposition 21]]
