---
id: obj_166
title: "Theorem 25"
types:
  - theorem
page_start: 71
page_end: 71
parent_id: "obj_165"
children_ids:
  - obj_167
sibling_ids:
  - obj_168
prerequisites: []
used_in:
  - obj_165
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 25

## Conceptual overview
This theorem characterizes the global optimum of the GAN objective function. It proves that the best possible result for the generator occurs when its distribution perfectly matches the data distribution, yielding a specific lower bound on the value function.

## Why it matters
It mathematically justifies the GAN framework by showing that if the networks are expressive enough, the training process should ideally converge to the true data distribution.

## Active recall
> [!question]- What is the optimal discriminator $D^*$ for a fixed generator $G$?
> $D^*(x) = \frac{\rho_X(x)}{\rho_X(x) + \rho_G(x)}$, where $\rho_G$ is the pushforward measure of the generator.

## Exact statement
For all $G$, $V(D^*, G) \geqslant -\log(4)$ and we have equality if $\rho_G = \rho_X$.

## Links
**Parent:** [[obj_165 - 3.5 Generative adversarial networks]]

**Children:**
- [[obj_167 - Proof of Theorem 25]]

**Used in:**
- [[obj_165 - 3.5 Generative adversarial networks]]
