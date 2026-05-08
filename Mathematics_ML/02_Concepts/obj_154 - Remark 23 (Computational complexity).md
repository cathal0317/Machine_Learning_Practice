---
id: obj_154
title: "Remark 23 (Computational complexity)"
types:
  - remark
page_start: 64
page_end: 64
parent_id: "obj_143"
children_ids: []
sibling_ids:
  - obj_142
  - obj_144
  - obj_145
  - obj_151
  - obj_155
prerequisites:
  - obj_145
  - obj_151
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 23 (Computational complexity)

## Conceptual overview
Explicitly compares the costs of computing the Jacobian matrix. Forward mode costs $O(s)$ (number of inputs), while reverse mode costs $O(t)$ (number of outputs).

## Why it matters
Explains why 'backpropagation' is the only viable method for training deep networks with millions of parameters.

## Active recall
> [!question]- Under what condition is reverse mode more efficient than forward mode?
> When the number of inputs $s$ is greater than the number of outputs $t$.

## Mental picture
Choose forward mode if you have a few inputs and many outputs; choose reverse mode if you have many inputs and a few outputs (like a loss function).

## Links
**Parent:** [[obj_143 - 3.3 Automatic differentiation]]

**Prerequisites:**
- [[obj_145 - 3.3.3 Forward mode]]
- [[obj_151 - 3.3.4 Reverse mode]]
