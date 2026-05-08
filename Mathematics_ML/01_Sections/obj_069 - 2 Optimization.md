---
id: obj_069
title: "2 Optimization"
types:
  - section
page_start: 31
page_end: 54
parent_id: "mod_01"
children_ids:
  - obj_068
  - obj_085
  - obj_099
  - obj_110
  - obj_127
sibling_ids:
  - obj_001
  - obj_002
  - obj_006
  - obj_134
prerequisites:
  - obj_006
used_in:
  - obj_134
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# 2 Optimization

## Conceptual overview
Optimization is the engine of machine learning. This section covers the theoretical foundations (convexity and duality) and the algorithmic implementations (gradient descent and stochastic gradient descent) used to minimize loss functions and find optimal model parameters.

## Why it matters
Without optimization theory, we could not efficiently train large-scale models or understand the convergence properties of our learning algorithms.

## Active recall
> [!question]- What are the three main sub-themes of this optimization section?
> 1. Convexity and optimality conditions; 2. Constrained optimization and Duality; 3. Iterative methods (Gradient Descent and SGD).

## Narrative flow
The section builds from simple minimizer definitions and convexity to more advanced topics like Lagrange duality and KKT conditions. It then transitions into algorithmic analysis, proving convergence rates for Gradient Descent and its stochastic variant.

## Links
**Parent:** [[mod_01 - MA3K1 - Mathematics of machine learning]]

**Children:**
- [[obj_068 - 2.1 Preliminaries]]
- [[obj_085 - 2.2 Constrained optimization]]
- [[obj_099 - 2.3 Support vector machines]]
- [[obj_110 - 2.4 Gradient descent]]
- [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_006 - 1 Statistical learning]]

**Used in:**
- [[obj_134 - Chapter 3 Neural networks]]
