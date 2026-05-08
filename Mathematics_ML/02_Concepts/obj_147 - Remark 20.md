---
id: obj_147
title: "Remark 20"
types:
  - remark
page_start: 61
page_end: 61
parent_id: "obj_145"
children_ids: []
sibling_ids:
  - obj_146
  - obj_149
prerequisites:
  - obj_145
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 20

## Conceptual overview
Points out that while evaluating a Jacobian-vector product costs the same as evaluating the function, computing the full Jacobian requires $O(s)$ passes, where $s$ is the input dimension.

## Why it matters
Highlights the efficiency limitations of forward mode for functions with many inputs and few outputs.

## Active recall
> [!question]- What is the cost of computing the full Jacobian using forward mode?
> $O(s)$, assuming the cost of evaluating the function once is $O(1)$ and $s$ is the number of inputs.

## Mental picture
Forward mode is inefficient for neural networks because there are millions of parameters (inputs) but only one loss value (output).

## Common confusions
Assuming automatic differentiation is always faster than other methods regardless of input/output dimensions.

## Links
**Parent:** [[obj_145 - 3.3.3 Forward mode]]

**Prerequisites:**
- [[obj_145 - 3.3.3 Forward mode]]
