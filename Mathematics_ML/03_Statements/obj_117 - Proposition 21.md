---
id: obj_117
title: "Proposition 21"
types:
  - proposition
page_start: 49
page_end: 49
parent_id: "obj_118"
children_ids:
  - obj_119
sibling_ids:
  - obj_116
  - obj_120
prerequisites:
  - obj_116
used_in:
  - obj_121
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 21

## Conceptual overview
This proposition translates the abstract definitions of strong convexity and smoothness into concrete quadratic inequalities. It shows that if a function has these properties, its values can be sandwiched between two parabolas at every point.

## Why it matters
These quadratic bounds are the primary tools used to prove convergence rates, as they allow us to quantify the progress made in each step of gradient descent.

## Active recall
> [!question]- What is the lower bound provided by $\mu$-strong convexity in Proposition 21?
> The lower bound is $f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2}\|x - y\|^2$.

> [!question]- What is the upper bound provided by $L$-Lipschitz smoothness in Proposition 21?
> The upper bound is $f(y) \le f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|x - y\|^2$.

## Exact statement
Let $f \in C^1(\mathbb{R}^p; \mathbb{R})$. If $f$ satisfies $(C_\mu)$, then for all $x, y \in \mathbb{R}^p$, $f(y) \ge f(x) + \langle \nabla f(x), y - x \rangle + \frac{\mu}{2}\|x - y\|^2$. If $f$ satisfies $(C_L)$, then for all $x, y \in \mathbb{R}^p$, $f(y) \le f(x) + \langle \nabla f(x), y - x \rangle + \frac{L}{2}\|x - y\|^2$.

## Links
**Parent:** [[obj_118 - 2.4.2 Strong convexity and smoothness]]

**Children:**
- [[obj_119 - Proof of Proposition 21]]

**Prerequisites:**
- [[obj_116 - Definition 10 (Strong convexity and smoothness)]]

**Used in:**
- [[obj_121 - Theorem 22]]
