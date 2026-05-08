---
id: obj_140
title: "Theorem 24 (The Universal Approximation theorem)"
types:
  - theorem
page_start: 57
page_end: 57
parent_id: "obj_138"
children_ids: []
sibling_ids:
  - obj_139
prerequisites:
  - obj_138
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 24 (The Universal Approximation theorem)

## Conceptual overview
Establishes that for any continuous function $f$ on a compact set and any precision $\varepsilon$, there exists a neural network with a single hidden layer that stays within $\varepsilon$ of $f$ everywhere.

## Why it matters
It guarantees that neural networks are sufficiently powerful to learn any continuous mapping, provided they have enough neurons.

## Active recall
> [!question]- What are the conditions on the activation function $\sigma$ in Theorem 24?
> It must be non-decreasing, continuous, and satisfy $\lim_{r \to \infty} \sigma(r) = 1$ and $\lim_{r \to -\infty} \sigma(r) = 0$.

> [!question]- Does Theorem 24 guarantee that we can find the required weights efficiently?
> No, it only guarantees their existence, not an efficient algorithm for finding them (though in practice we use SGD).

## Exact statement
Suppose that $\sigma$ satisfies 
$$
\lim_{r \to \infty} \sigma(r) = 1 \quad \text{and} \quad \lim_{r \to -\infty} \sigma(r) = 0.
$$
 For any compact set $\Omega \subset \mathbb{R}^p$, the space spanned by the functions 
$$
\mathcal{F}_\sigma := \{ \phi_{w,b}(x) := \sigma(x^\top w + b) ; w \in \mathbb{R}^p, b \in \mathbb{R} \}
$$
 is dense in $C(\Omega)$ with uniform convergence. This means that for any continuous function $f : \Omega \to \mathbb{R}$ and any $\varepsilon > 0$, there exists $q \in \mathbb{N}$, $w_j \in \mathbb{R}^p$ and $a_j, b_j \in \mathbb{R}$ such that 
$$
\sup_{x \in \Omega} | f(x) - \sum_{k=1}^q a_k \phi_{w_k, b_k}(x) | \le \varepsilon.
$$

## Links
**Parent:** [[obj_138 - 3.1.2 Universal approximation]]

**Prerequisites:**
- [[obj_138 - 3.1.2 Universal approximation]]
