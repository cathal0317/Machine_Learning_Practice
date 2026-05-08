---
id: obj_172
title: "Example (Principle component analysis)"
types:
  - example
page_start: 74
page_end: 74
parent_id: "obj_169"
children_ids: []
sibling_ids:
  - obj_171
prerequisites:
  - obj_171
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example (Principle component analysis)

## Conceptual overview
This example frames Principal Component Analysis (PCA) as a specific instance of an autoencoder where the encoder and decoder are linear orthogonal mappings.

## Why it matters
It demonstrates that the classic statistical technique of PCA is fundamentally equivalent to finding the best linear reconstruction of data.

## Active recall
> [!question]- In the autoencoder framework, what determines the optimal matrix $Q$ for PCA?
> The optimal $Q$ consists of the $d$ largest orthonormal eigenvectors of $XX^\top$.

## When to use
Used for simple, linear dimensionality reduction when reconstruction error in the Euclidean norm needs to be minimized.

## Core pattern
Solving an autoencoder problem with the constraint that maps are linear and $Q^\top Q = Id$.

## Links
**Parent:** [[obj_169 - 3.6.1 Dimension reduction]]

**Prerequisites:**
- [[obj_171 - Autoencoders]]
