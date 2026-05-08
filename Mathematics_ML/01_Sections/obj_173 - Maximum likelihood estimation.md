---
id: obj_173
title: "Maximum likelihood estimation"
types:
  - discussion
page_start: 74
page_end: 75
parent_id: "obj_174"
children_ids: []
sibling_ids:
  - obj_175
  - obj_176
  - obj_178
  - obj_180
  - obj_181
prerequisites: []
used_in:
  - obj_174
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Maximum likelihood estimation

## Conceptual overview
Maximum Likelihood Estimation (MLE) in latent models seeks to find parameters $\theta$ that maximize the marginal probability of the data, which requires integrating over all possible latent variables.

## Why it matters
It identifies the core difficulty in generative modeling: the marginal likelihood integral is often intractable, necessitating approximate inference methods like the ELBO.

## Active recall
> [!question]- Why is the direct MLE objective $p_\theta(x) = \int p_\theta(x|z)p_Z(z)dz$ difficult to optimize?
> It is intractable because we need to compute an integral over all possible latent states $z$, which requires evaluating the mapping at every point.

## Narrative flow
Defines the MLE goal for VAEs, points out the computational bottleneck of the integral, and transitions to the evidence lower bound as the solution.

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Used in:**
- [[obj_174 - 3.6.2 Variational autoencoders]]
