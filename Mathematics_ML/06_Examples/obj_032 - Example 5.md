---
id: obj_032
title: "Example 5"
types:
  - example
page_start: 20
page_end: 20
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_033
  - obj_034
  - obj_036
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites:
  - obj_034
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 5

## Conceptual overview
This example calculates the tail probability $\mathbb{P}(W \ge t)$ for $W \sim \mathcal{N}(0, \sigma^2)$. By substituting the Gaussian MGF into the Chernoff bound and optimizing the parameter $\alpha$, it recovers the standard sub-Gaussian tail bound.

## Why it matters
It serves as the prototypical calculation for all sub-Gaussian concentration results used in the course.

## Active recall
> [!question]- What value of $\alpha$ achieves the infimum in the Chernoff bound for a $\mathcal{N}(0, \sigma^2)$ variable?
> The infimum is achieved at $\alpha = t/\sigma^2$.

## When to use
When you need to derive or justify the exponential decay of the tails of a Normal distribution.

## Core pattern
Insert the specific MGF of a distribution into the general Chernoff bound formula and solve the resulting calculus problem to find the tightest bound.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Prerequisites:**
- [[obj_034 - The Chernoff bound]]
