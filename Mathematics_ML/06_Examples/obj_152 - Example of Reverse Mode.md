---
id: obj_152
title: "Example of Reverse Mode"
types:
  - example
page_start: 63
page_end: 63
parent_id: "obj_151"
children_ids: []
sibling_ids:
  - obj_153
prerequisites:
  - obj_151
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example of Reverse Mode

## Conceptual overview
Demonstrates calculating the Jacobian of a function with two inputs and two outputs using a forward pass to get intermediate values followed by a backward pass to get gradients.

## Why it matters
Provides a numerical walkthrough that clarifies the abstract recursive formula of reverse mode AD.

## Active recall
> [!question]- How is the forward pass used in reverse mode AD?
> It is used to compute and store the intermediate values (like $z_3 = \log(2)$) which are needed to evaluate the partial derivatives $\partial z_m / \partial z_k$ during the backward pass.

## When to use
When verifying the manual execution of the backpropagation algorithm.

## Core pattern
Storing forward values to compute local gradients, then accumulating these gradients in reverse topological order.

## Links
**Parent:** [[obj_151 - 3.3.4 Reverse mode]]

**Prerequisites:**
- [[obj_151 - 3.3.4 Reverse mode]]
