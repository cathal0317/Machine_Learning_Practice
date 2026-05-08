---
id: obj_146
title: "Remark 19"
types:
  - remark
page_start: 61
page_end: 61
parent_id: "obj_145"
children_ids: []
sibling_ids:
  - obj_147
  - obj_149
prerequisites:
  - obj_145
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 19

## Conceptual overview
Explains that by initializing the input derivatives with a vector $r$, the forward mode procedure computes the product of the Jacobian matrix and that vector.

## Why it matters
Enables the computation of directional derivatives in a single pass.

## Active recall
> [!question]- What does each $\dot{z}_k$ represent if inputs are initialized with $\dot{z}_{-i} = r_i$?
> It represents the directional derivative $r^\top \nabla z_k$.

## Mental picture
Instead of just tracking how a node changes with one axis, you're tracking how it changes along a specific direction in the input space.

## Links
**Parent:** [[obj_145 - 3.3.3 Forward mode]]

**Prerequisites:**
- [[obj_145 - 3.3.3 Forward mode]]
