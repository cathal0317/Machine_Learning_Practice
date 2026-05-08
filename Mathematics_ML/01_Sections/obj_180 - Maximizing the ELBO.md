---
id: obj_180
title: "Maximizing the ELBO"
types:
  - discussion
page_start: 76
page_end: 76
parent_id: "obj_174"
children_ids: []
sibling_ids:
  - obj_173
  - obj_175
  - obj_176
  - obj_178
  - obj_181
prerequisites:
  - obj_175
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Maximizing the ELBO

## Conceptual overview
To maximize the ELBO in practice, one models the distributions $q_\psi$ and $p_\theta$ as Gaussians parameterized by neural networks and optimizes the parameters via stochastic gradient descent.

## Why it matters
It explains how the theoretical bound is converted into an executable training loop on a finite dataset.

## Active recall
> [!question]- What function is typically used to model the parameters of the Gaussian conditional $p_\theta(\cdot|z)$?
> A deterministic function $G_\theta$ (a neural network) is used to map the latent variable $z$ to the mean of the data distribution.

## Narrative flow
Summarizes the modeling assumptions (Gaussian encoder/decoder), defines the empirical objective function $\hat{\mathcal{L}}(\psi, \theta)$, and prepares the ground for the 'computation of gradients'.

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Prerequisites:**
- [[obj_175 - Proposition 26 (ELBO)]]
