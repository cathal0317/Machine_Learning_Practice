---
id: obj_131
title: "Remark 16"
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
  - obj_132
prerequisites:
  - obj_127
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 16

## Conceptual overview
This remark contrasts the theoretical slower rate of SGD ($O(1/k)$) with its practical superiority for large datasets. It explains that SGD can make substantial initial progress without ever having to process all data points, unlike GD which requires a full pass just for the first step.

## Why it matters
It resolves the apparent paradox of why we use an algorithm that is theoretically 'slower' (sublinear vs linear) in practice.

## Active recall
> [!question]- Why is SGD often preferred over GD in practice even if its theoretical convergence rate is slower?
> Because SGD iterations are much cheaper, allowing the algorithm to make significant progress toward the optimum long before GD has even finished a single full pass through a large dataset.

## Mental picture
Imagine GD as a team that needs to read every page of a thousand-page book before they can take a single step forward. SGD is a single person who reads one random page and immediately takes a step based on what they learned. In the time it takes the team to move once, the individual has already moved hundreds of times.

## Common confusions
Thinking that linear convergence always beats sublinear convergence in time; this ignores the fact that iteration costs differ by a factor of $n$.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_127 - 2.5 Stochastic gradient descent]]
