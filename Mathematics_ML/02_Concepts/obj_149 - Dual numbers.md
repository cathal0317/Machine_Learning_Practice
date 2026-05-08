---
id: obj_149
title: "Dual numbers"
types:
  - concept
page_start: 61
page_end: 62
parent_id: "obj_145"
children_ids:
  - obj_148
  - obj_150
sibling_ids:
  - obj_146
  - obj_147
prerequisites: []
used_in:
  - obj_150
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Dual numbers

## Conceptual overview
An algebra consisting of elements $x + \varepsilon x'$ where $\varepsilon^2 = 0$. When a function is applied to a dual number, the first term yields the function value and the coefficient of $\varepsilon$ yields the derivative.

## Why it matters
Allows for the implementation of exact differentiation through standard code via operator overloading.

## Active recall
> [!question]- What is the result of multiplying $(a + b\varepsilon)$ and $(c + d\varepsilon)$?
> $ac + (ad + bc)\varepsilon$, since the $bd\varepsilon^2$ term is zero.

> [!question]- How is the application of a function $f$ to a dual number defined?
> $f(a + b\varepsilon) := f(a) + f'(a)b\varepsilon$.

## Mental picture
A dual number is a number that carries its own 'local sensitivity' (derivative) along with it as it goes through arithmetic operations.

## Common confusions
Thinking that $\varepsilon$ is an infinitesimal constant rather than a formal algebraic symbol.

## Links
**Parent:** [[obj_145 - 3.3.3 Forward mode]]

**Children:**
- [[obj_148 - Remark 21]]
- [[obj_150 - Example of Dual Numbers]]

**Used in:**
- [[obj_150 - Example of Dual Numbers]]
