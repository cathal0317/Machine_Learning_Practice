---
id: obj_108
title: "Gradient descent algorithm"
types:
  - algorithm
page_start: 46
page_end: 46
parent_id: "obj_110"
children_ids: []
sibling_ids:
  - obj_109
  - obj_111
  - obj_112
  - obj_118
  - obj_122
prerequisites: []
used_in:
  - obj_110
  - obj_112
  - obj_126
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Gradient descent algorithm

## Conceptual overview
Gradient descent is a first-order iterative optimization algorithm. It starts from an initial guess and repeatedly updates the parameters by subtracting the gradient of the function scaled by a stepsize $\tau_k$, based on the fact that the negative gradient points in the direction of local steepest decrease.

## Why it matters
It is the workhorse of optimization in machine learning, used to train almost every modern model including deep neural networks.

## Active recall
> [!question]- State the update rule for the gradient descent algorithm.
> The update rule is $w_{k+1} = w_k - \tau_k \nabla f(w_k)$.

## When to use
Use this algorithm when the objective function is differentiable and you need to find a local minimum iteratively.

## Core pattern
Take current position, compute gradient, step in the opposite direction, repeat until convergence.

## Links
**Parent:** [[obj_110 - 2.4 Gradient descent]]

**Used in:**
- [[obj_110 - 2.4 Gradient descent]]
- [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]
- [[obj_126 - Stochastic gradient descent (SGD) Algorithm]]
