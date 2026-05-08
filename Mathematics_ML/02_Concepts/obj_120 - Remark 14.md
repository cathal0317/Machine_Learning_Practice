---
id: obj_120
title: "Remark 14"
types:
  - remark
page_start: 50
page_end: 50
parent_id: "obj_118"
children_ids: []
sibling_ids:
  - obj_116
  - obj_117
prerequisites:
  - obj_116
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 14

## Conceptual overview
This remark provides an alternative characterization of strong convexity and smoothness for twice-differentiable functions. It states that these properties are equivalent to global upper and lower bounds on the eigenvalues of the function's Hessian matrix.

## Why it matters
Checking eigenvalues of a Hessian is often easier and more intuitive than checking the gradient conditions across the entire domain.

## Active recall
> [!question]- For a twice-differentiable function, how is $\mu$-strong convexity expressed in terms of the Hessian?
> It is equivalent to the condition $\mu I \preceq \nabla^2 f(x)$ for all $x$, meaning all eigenvalues of the Hessian are at least $\mu$.

> [!question]- What is the Hessian-based equivalent for $L$-Lipschitz smoothness?
> It is equivalent to the condition $\nabla^2 f(x) \preceq L I$ for all $x$, meaning all eigenvalues of the Hessian are at most $L$.

## Mental picture
Think of the Hessian as measuring the curvature of the 'floor' of your objective function. Remark 14 says that smoothness means the floor never gets too steep or curvy, while strong convexity means the floor is always 'cupped' enough that it doesn't become flat.

## Common confusions
Confusing 'positive definite' (convex) with 'strongly convex'. A positive definite Hessian only ensures convexity; to be $\mu$-strongly convex, there must be a uniform strictly positive lower bound on the eigenvalues.

## Links
**Parent:** [[obj_118 - 2.4.2 Strong convexity and smoothness]]

**Prerequisites:**
- [[obj_116 - Definition 10 (Strong convexity and smoothness)]]
