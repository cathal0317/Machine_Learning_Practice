---
id: obj_136
title: "Example 18"
types:
  - example
page_start: 56
page_end: 56
parent_id: "obj_133"
children_ids: []
sibling_ids:
  - obj_135
  - obj_137
  - obj_138
prerequisites:
  - obj_133
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 18

## Conceptual overview
This example provides a concrete construction of a two-layer neural network with ReLU activation that can correctly classify the non-linearly separable XOR function. It uses a hidden layer of two neurons to create a representation that is separable by a linear output layer.

## Why it matters
It historically illustrates the power of multi-layer networks to solve problems that single-layer linear perceptrons cannot, justifying the move to deep architectures.

## Active recall
> [!question]- Why can't the XOR function be solved by a linear separator?
> Because the points (0,1) and (1,0) cannot be separated from (0,0) and (1,1) by a single line in 2D space.

> [!question]- What activation function is used in the network constructed in Example 18?
> The ReLU activation function, denoted as $(x)_+ = \max(0, x)$.

## When to use
When demonstrating the necessity of hidden layers and non-linearities for solving non-linearly separable classification problems.

## Core pattern
Composing two linear separators via a non-linear hidden layer to approximate a target function.

## Links
**Parent:** [[obj_133 - 3.1 Multilayer perceptrons]]

**Prerequisites:**
- [[obj_133 - 3.1 Multilayer perceptrons]]
