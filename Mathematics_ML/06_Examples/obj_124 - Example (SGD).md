---
id: obj_124
title: "Example (SGD)"
types:
  - example
page_start: 52
page_end: 52
parent_id: "obj_127"
children_ids: []
sibling_ids:
  - obj_125
  - obj_126
  - obj_128
  - obj_129
  - obj_131
  - obj_132
prerequisites:
  - obj_127
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example (SGD)

## Conceptual overview
This example compares the computational complexity of one step of Stochastic Gradient Descent (SGD) versus full Gradient Descent (GD) for a sum-of-squares objective. It highlights that SGD is significantly cheaper per iteration when the number of data points is large.

## Why it matters
It explains the practical motivation for SGD: when $n$ is very large, the cost of a single full gradient calculation is prohibitive, and a noisy estimate can be more efficient.

## Active recall
> [!question]- According to the example, what is the cost of one iteration of SGD compared to GD for $n$ data points in $p$ dimensions?
> One iteration of SGD costs $O(p)$, whereas one iteration of full GD costs $O(np)$, making SGD $n$ times cheaper.

## When to use
Use when comparing optimization algorithms for large-scale datasets where the bottleneck is the evaluation of the full objective gradient.

## Core pattern
Assess the complexity of evaluating a single component gradient versus the sum of all component gradients.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_127 - 2.5 Stochastic gradient descent]]
