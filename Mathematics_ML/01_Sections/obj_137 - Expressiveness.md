---
id: obj_137
title: "Expressiveness"
types:
  - discussion
page_start: 56
page_end: 56
parent_id: "obj_133"
children_ids: []
sibling_ids:
  - obj_135
  - obj_136
  - obj_138
prerequisites:
  - obj_133
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Expressiveness

## Conceptual overview
Explores how the representational power of MLPs depends on the activation function and the number of neurons. It notes that linear activation functions prevent any increase in expressiveness regardless of network size, while non-polynomial activations are required for universal approximation.

## Why it matters
It explains the fundamental motivation for using non-linear activation functions in deep learning.

## Active recall
> [!question]- Why can't $\sigma$ be a linear function if we want high expressivity?
> Because if $\sigma$ is linear, the composition $h_{w,a,b}(x)$ is still a linear function in $x$, meaning the network cannot learn non-linear patterns.

> [!question]- What happens if the activation function $\sigma$ is a polynomial of degree $d$?
> The resulting network $h_{w,a,b}(x)$ will always be a polynomial of degree $d$, limiting its capacity to approximate non-polynomial functions.

## Narrative flow
The discussion starts by wanting to increase complexity with the number of neurons $q$. It then shows that linear or polynomial activation functions severely limit this complexity, motivating the need for standard non-linear activations like sigmoids or ReLU.

## Links
**Parent:** [[obj_133 - 3.1 Multilayer perceptrons]]

**Prerequisites:**
- [[obj_133 - 3.1 Multilayer perceptrons]]
