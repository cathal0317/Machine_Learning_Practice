---
id: obj_171
title: "Autoencoders"
types:
  - concept
page_start: 74
page_end: 74
parent_id: "obj_169"
children_ids: []
sibling_ids:
  - obj_172
prerequisites: []
used_in:
  - obj_172
  - obj_174
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Autoencoders

## Conceptual overview
An autoencoder consists of two mappings: an encoder $z = E(x)$ which compresses data to a lower-dimensional representation, and a decoder $x' = D(z)$ which attempts to reconstruct the original data.

## Why it matters
It is a fundamental unsupervised learning structure used to extract features and compress data into a bottleneck representation.

## Active recall
> [!question]- What is the optimization problem for training a basic autoencoder?
> $\min_{E,D} \frac{1}{n} \sum_{i=1}^n ||D(E(x_i)) - x_i||^2$.

## Mental picture
A funnel where information is squeezed through a tiny neck (the latent space) and must be carefully unfolded back to its original size, forcing the neck to only pass the most important essence of the information.

## Common confusions
Confusing the objective; the goal is not just any mapping, but specifically reconstruction that forces the latent space to be a meaningful summary of the input.

## Links
**Parent:** [[obj_169 - 3.6.1 Dimension reduction]]

**Used in:**
- [[obj_172 - Example (Principle component analysis)]]
- [[obj_174 - 3.6.2 Variational autoencoders]]
