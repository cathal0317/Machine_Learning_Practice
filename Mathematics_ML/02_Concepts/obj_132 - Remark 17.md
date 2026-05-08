---
id: obj_132
title: "Remark 17"
types:
  - remark
page_start: 54
page_end: 54
parent_id: "obj_127"
children_ids: []
sibling_ids:
  - obj_124
  - obj_125
  - obj_126
  - obj_128
  - obj_129
  - obj_131
prerequisites:
  - obj_129
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 17

## Conceptual overview
This remark notes that if the objective function is convex with a Lipschitz gradient but lacks strong convexity, the convergence rate of SGD slows further to $O(1/\sqrt{k})$.

## Why it matters
It defines the performance expectations for SGD when applied to objectives that are 'flatter' than parabolas, such as those found in many deep learning layers.

## Active recall
> [!question]- What is the expected convergence rate for SGD on a smooth, convex (but not strongly convex) objective?
> The expected objective gap $\mathbb{E}(f(w_k)) - \min f(w)$ converges at a rate of $O(1/\sqrt{k})$.

## Mental picture
In a non-strongly convex setting, the objective can have extremely flat regions. The algorithm struggles more to pinpoint the exact minimum because the gradients provide less and less signal as it approaches the optimum.

## Common confusions
Confusing the iterate convergence norm with objective value convergence; in the non-strongly convex case, we usually only provide bounds for the objective value.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_129 - Theorem 23]]
