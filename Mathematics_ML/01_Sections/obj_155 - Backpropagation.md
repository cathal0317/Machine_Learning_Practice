---
id: obj_155
title: "Backpropagation"
types:
  - discussion
page_start: 64
page_end: 65
parent_id: "obj_143"
children_ids: []
sibling_ids:
  - obj_142
  - obj_144
  - obj_145
  - obj_151
  - obj_154
prerequisites:
  - obj_151
used_in:
  - obj_183
analogous_to: []
same_pattern_as:
  - obj_151
family: "section-family"
---

# Backpropagation

## Conceptual overview
Applies the theory of reverse mode automatic differentiation to the specific case of a feedforward neural network and a scalar loss function. It derives the concrete update rules for layer weights and activations.

## Why it matters
Backpropagation is the fundamental algorithm for training almost all modern neural networks.

## Active recall
> [!question]- What are the recursive variables in the backpropagation equations?
> The layer-wise sensitivities $\dot{z}_i^m = \sigma'(z_i^m)\dot{a}_i^m$ and $\dot{a}_i^{m-1} = \sum_k \dot{z}_k^m W_{k,i}^m$.

## Narrative flow
Defines the specific feedforward network equations, then specializes reverse mode AD to compute gradients of the loss with respect to all weights and biases in the network.

## Links
**Parent:** [[obj_143 - 3.3 Automatic differentiation]]

**Prerequisites:**
- [[obj_151 - 3.3.4 Reverse mode]]

**Used in:**
- [[obj_183 - Monte-carlo approximations]]

**Same pattern as:**
- [[obj_151 - 3.3.4 Reverse mode]]
