---
id: obj_183
title: "Monte-carlo approximations"
types:
  - technique
page_start: 77
page_end: 78
parent_id: "obj_181"
children_ids: []
sibling_ids:
  - obj_179
prerequisites:
  - obj_181
  - obj_155
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Monte-carlo approximations

## Conceptual overview
Monte-Carlo approximation handles the expectation in the ELBO reconstruction term by taking a single sample. To allow gradients to flow, the 'reparameterization trick' expresses the sample as a deterministic function of parameters and independent noise.

## Why it matters
It is the standard solution for the gradient estimation problem in VAEs, allowing them to be trained using efficient backpropagation.

## Active recall
> [!question]- How is the stochastic gradient with respect to $\psi$ evaluated after reparameterization?
> Since $z$ is now a deterministic function of $\psi$, the gradient $\partial_\psi$ can be evaluated directly via backpropagation.

## When to use
Used when training stochastic computation graphs where expectations need to be differentiated.

## Core pattern
Reparameterization: moving the stochasticity of a node into an independent input to restore differentiability.

## Links
**Parent:** [[obj_181 - Computation of gradients]]

**Prerequisites:**
- [[obj_181 - Computation of gradients]]
- [[obj_155 - Backpropagation]]
