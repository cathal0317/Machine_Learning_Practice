---
id: obj_156
title: "Definition 11 (Convolution)"
types:
  - definition
page_start: 65
page_end: 65
parent_id: "obj_158"
children_ids: []
sibling_ids:
  - obj_157
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

# Definition 11 (Convolution)

## Conceptual overview
Defines the convolution of two functions $f$ and $h$ on $\mathbb{R}^d$. It represents an integral of the product of the two functions after one is reversed and shifted.

## Why it matters
Convolution is the fundamental operation in signal processing and CNNs, providing a way to perform localized, translation-invariant transformations.

## Active recall
> [!question]- Give the mathematical expression for the convolution of $f$ and $h$.
> $g(t) := (f \star h)(t) := \int f(x)h(t - x)dx$.

## Mental picture
Think of one function as a 'filter' that is flipped and slid across the other function, measuring the local overlap at each position.

## Common confusions
Confusing convolution with a simple element-wise product; convolution involves a sum/integral over a range of shifted values.

## Links
**Parent:** [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

**Used in:**
- [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]
