---
id: obj_162
title: "Residual neural networks"
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
  - obj_161
prerequisites:
  - obj_158
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Residual neural networks

## Conceptual overview
Residual networks (ResNets) utilize skip layer connections that allow the network to learn residual mappings $F(x) = H(x) - x$ rather than the direct mapping $H(x)$. This is achieved by adding the input $x_k$ back to the output of a layer $F_k(x_k)$.

## Why it matters
They mitigate the vanishing gradient problem in very deep networks, enabling the effective training of architectures with hundreds or thousands of layers.

## Active recall
> [!question]- What is the formula for a skip connection in a ResNet?
> The connection takes the form $x_{k+1} = x_k + F_k(x_k)$, where $F_k$ is the residual learned by the layer.

> [!question]- To what numerical method can ResNet skip connections be compared?
> Adding an artificial time step makes them similar to an explicit Euler discretization of an ordinary differential equation (ODE): $\dot{x}_t = F_t(x_t)$.

## Mental picture
Imagine a highway with exit ramps that loop immediately back onto the main road; the 'skip connections' allow signal information to bypass specific processing layers, ensuring the core signal isn't lost as the depth increases.

## Common confusions
Confusing ResNets with simple feedforward paths; the critical difference is the additive identity mapping shortcut that forces the non-linear layers to learn 'changes' rather than the full target.

## Links
**Parent:** [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

**Prerequisites:**
- [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]
