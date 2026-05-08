---
id: obj_126
title: "Stochastic gradient descent (SGD) Algorithm"
types:
  - algorithm
page_start: 52
page_end: 52
parent_id: "obj_127"
children_ids: []
sibling_ids:
  - obj_124
  - obj_125
  - obj_128
  - obj_129
  - obj_131
  - obj_132
prerequisites:
  - obj_108
used_in:
  - obj_127
  - obj_129
analogous_to:
  - obj_108
same_pattern_as: []
family: "technique-family"
---

# Stochastic gradient descent (SGD) Algorithm

## Conceptual overview
The SGD algorithm is an iterative optimization method that updates the parameter vector by moving in the direction of the negative gradient of a single, randomly selected component function. This provides a computationally efficient alternative to standard gradient descent when the total number of components is large.

## Why it matters
SGD is the workhorse of modern deep learning, enabling the training of large models on massive datasets.

## Active recall
> [!question]- What is the update rule for the SGD algorithm?
> The update rule is $w_{k+1} = w_k - \tau_k \nabla f_{i_k}(w_k)$, where $i_k$ is sampled uniformly at random from $\{1, ..., n\}$.

> [!question]- Why are the iterates $w_k$ in SGD considered random vectors?
> They are random because each update depends on the random index $i_k$ chosen at each iteration step $k$.

## When to use
Use for minimizing objective functions that are expressed as a sum of many component functions, especially in large-scale machine learning.

## Core pattern
Replace the full gradient calculation $\frac{1}{n} \sum \nabla f_i(w)$ with a single randomly chosen component gradient $\nabla f_{i_k}(w)$.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_108 - Gradient descent algorithm]]

**Used in:**
- [[obj_127 - 2.5 Stochastic gradient descent]]
- [[obj_129 - Theorem 23]]

**Analogous to:**
- [[obj_108 - Gradient descent algorithm]]
