---
id: obj_161
title: "Max-pooling"
types:
  - concept
page_start: 68
page_end: 68
parent_id: "obj_158"
children_ids: []
sibling_ids:
  - obj_156
  - obj_157
  - obj_160
  - obj_162
prerequisites:
  - obj_158
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Max-pooling

## Conceptual overview
Max-pooling is a non-linear downsampling technique used in CNNs to reduce the spatial dimensions of feature maps. It slides a window over the input and outputs the maximum value from each region, effectively performing a form of non-linear subsampling.

## Why it matters
It reduces the computational complexity of subsequent layers, manages overfitting, and provides a degree of local translation invariance by summarizing prominent features within regions.

## Active recall
> [!question]- How does max-pooling help neurons in deeper layers of a CNN?
> By reducing the spatial resolution, max-pooling allows neurons in deeper layers to be dependent on larger and larger areas of the original image domain, capturing more elaborate shapes.

## Mental picture
Like looking at a photo through a coarse mesh where you only keep the single brightest pixel in each hole, shrinking the image while preserving the most distinct highlights.

## Common confusions
Thinking that max-pooling is a linear operation; it is actually non-linear because it chooses the maximum value rather than taking a linear combination.

## Links
**Parent:** [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

**Prerequisites:**
- [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]
