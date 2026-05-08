---
id: obj_157
title: "Definition 12 (Discrete convolution)"
types:
  - definition
page_start: 65
page_end: 65
parent_id: "obj_158"
children_ids: []
sibling_ids:
  - obj_156
  - obj_160
  - obj_161
  - obj_162
prerequisites: []
used_in:
  - obj_158
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 12 (Discrete convolution)

## Conceptual overview
Defines the convolution operation for discrete sequences $(h_k)$ and $(f_k)$. It uses a sum over products of shifted indices.

## Why it matters
Discrete convolution is the operation actually implemented in computer software and CNN hardware.

## Active recall
> [!question]- Define the discrete circular convolution of sequences $h$ and $f$.
> $g_k = \sum_{i=1}^n f_{k-i}h_i$.

## Mental picture
Like the continuous version, but replacing an integral with a sum over index-shifted elements.

## Links
**Parent:** [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

**Used in:**
- [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]
