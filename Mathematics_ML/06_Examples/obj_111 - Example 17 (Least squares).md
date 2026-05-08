---
id: obj_111
title: "Example 17 (Least squares)"
types:
  - example
page_start: 47
page_end: 47
parent_id: "obj_110"
children_ids: []
sibling_ids:
  - obj_108
  - obj_109
  - obj_112
  - obj_118
  - obj_122
prerequisites:
  - obj_077
  - obj_108
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 17 (Least squares)

## Conceptual overview
This example demonstrates how to compute an optimal adaptive stepsize for the gradient descent algorithm when applied to a standard quadratic least squares problem. By exploiting the specific structure of the quadratic objective, the stepsize can be calculated analytically at each iteration rather than being fixed or searched.

## Why it matters
It provides a concrete implementation of gradient descent where the learning rate is chosen optimally to minimize the function along the search direction, illustrating the 'greedy' selection strategy.

## Active recall
> [!question]- In the context of Example 17, how is the greedy stepsize $\tau_k$ calculated for the least squares objective?
> The stepsize is calculated using the formula $\tau_k = \|r_k\|^2 / \|Ar_k\|^2$, where $r_k = A^\top(Aw_k - b)$ is the gradient.

> [!question]- What is the geometric implication of choosing the greedy stepsize in this example?
> Choosing the greedy stepsize ensures that consecutive search directions are orthogonal ($\langle r_k, r_{k+1} \rangle = 0$), which often leads to a zig-zag trajectory toward the minimizer.

## When to use
Use when applying gradient descent to a quadratic objective $f(w) = \frac{1}{2}\|Aw - b\|^2$ where an analytical stepsize is preferred over fixed rates.

## Core pattern
Derive the stepsize by minimizing the function $h(\tau) = f(w_k - \tau \nabla f(w_k))$ analytically at each step.

## Links
**Parent:** [[obj_110 - 2.4 Gradient descent]]

**Prerequisites:**
- [[obj_077 - Example 13 (Least squares)]]
- [[obj_108 - Gradient descent algorithm]]
