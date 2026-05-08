---
id: obj_168
title: "Algorithm for GAN optimization"
types:
  - algorithm
page_start: 73
page_end: 73
parent_id: "obj_165"
children_ids: []
sibling_ids:
  - obj_166
prerequisites:
  - obj_165
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Algorithm for GAN optimization

## Conceptual overview
Training GANs involves an alternating optimization procedure where the discriminator is first updated to better distinguish real from fake, followed by an update to the generator to better fool the discriminator.

## Why it matters
Provides a practical way to solve the non-concave min-max optimization problem using standard stochastic gradient descent tools.

## Active recall
> [!question]- What are the two main steps in a single iteration of the GAN training algorithm?
> 1. Fix the generator and update the discriminator using $N$ steps of SGD to maximize the value function. 2. Fix the discriminator and update the generator using gradient descent to minimize the value function.

## When to use
Used during the training phase of a GAN to find the optimal parameters for both networks.

## Core pattern
Alternating optimization on a zero-sum game objective.

## Links
**Parent:** [[obj_165 - 3.5 Generative adversarial networks]]

**Prerequisites:**
- [[obj_165 - 3.5 Generative adversarial networks]]
