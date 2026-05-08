---
id: obj_125
title: "Remark 15"
types:
  - remark
page_start: 52
page_end: 52
parent_id: "obj_127"
children_ids: []
sibling_ids:
  - obj_124
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
family: "concept-family"
---

# Remark 15

## Conceptual overview
This remark explains that although we often apply SGD to a finite sum (the empirical risk), it is also a valid method for minimizing the expected risk directly if we can sample from the data distribution. It frames SGD as an algorithm that works with noisy, unbiased estimates of the true gradient.

## Why it matters
It bridges the gap between empirical risk minimization (optimizing on training data) and generalization (optimizing for the underlying distribution).

## Mental picture
Instead of having a fixed finite map of a mountain, imagine you are in the dark and can only feel the slope directly under your feet. By taking steps based on these local, 'noisy' slope readings, you can still find your way to the valley floor.

## Common confusions
A student might think SGD is just an approximation of GD; Remark 15 suggests that from a statistical learning perspective, it is actually targeting the true population risk directly.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_127 - 2.5 Stochastic gradient descent]]
