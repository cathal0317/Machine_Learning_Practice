---
id: obj_153
title: "Remark 22"
types:
  - remark
page_start: 63
page_end: 63
parent_id: "obj_151"
children_ids: []
sibling_ids:
  - obj_152
prerequisites:
  - obj_151
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 22

## Conceptual overview
Explains that reverse mode automatic differentiation computes the product of the transposed Jacobian and a vector $r$.

## Why it matters
Clarifies the exact mathematical operator computed by reverse mode, which is the dual of the operator computed by forward mode.

## Active recall
> [!question]- What quantity is computed at the end of the backward pass in Remark 22?
> The vector $(J^\top r)_i$, where $J$ is the Jacobian of the function.

## Mental picture
If forward mode is $Jr$, reverse mode is $J^\top r$.

## Common confusions
Thinking reverse mode AD computes the Jacobian matrix directly in one pass.

## Links
**Parent:** [[obj_151 - 3.3.4 Reverse mode]]

**Prerequisites:**
- [[obj_151 - 3.3.4 Reverse mode]]
