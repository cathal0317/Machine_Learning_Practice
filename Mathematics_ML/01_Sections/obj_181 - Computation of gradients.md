---
id: obj_181
title: "Computation of gradients"
types:
  - discussion
page_start: 76
page_end: 78
parent_id: "obj_174"
children_ids:
  - obj_179
  - obj_183
sibling_ids:
  - obj_173
  - obj_175
  - obj_176
  - obj_178
  - obj_180
prerequisites:
  - obj_179
used_in:
  - obj_183
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Computation of gradients

## Conceptual overview
Gradient computation for VAEs is split into two parts: an analytic part for the KL term and a stochastic approximation for the reconstruction term. The latter requires the reparameterization trick to move sampling outside the gradient path.

## Why it matters
Direct backpropagation through a random sample is impossible; these techniques make VAEs trainable via end-to-end gradient descent.

## Active recall
> [!question]- Why can't we use standard backpropagation directly through the sampling step $z \sim q_\psi$?
> Standard backpropagation requires deterministic differentiable nodes; the stochastic nature of sampling prevents gradients from flowing back to the parameters $\psi$ of the encoder.

## Narrative flow
Breaks the gradient problem into the KL term (solved by Lemma 27) and the reconstruction term (solved by Monte-Carlo approximations and the reparameterization trick).

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Children:**
- [[obj_179 - Lemma 27 (KL divergence between Gaussian distributions)]]
- [[obj_183 - Monte-carlo approximations]]

**Prerequisites:**
- [[obj_179 - Lemma 27 (KL divergence between Gaussian distributions)]]

**Used in:**
- [[obj_183 - Monte-carlo approximations]]
