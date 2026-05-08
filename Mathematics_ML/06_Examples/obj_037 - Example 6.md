---
id: obj_037
title: "Example 6"
types:
  - example
page_start: 21
page_end: 21
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_032
  - obj_033
  - obj_034
  - obj_036
  - obj_038
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites:
  - obj_036
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 6

## Conceptual overview
This example shows that a Rademacher variable $\varepsilon$ taking values $\pm 1$ satisfies the sub-Gaussian condition with $\sigma=1$. It involves comparing the power series expansion of $\cosh(\alpha)$ to that of $\exp(\alpha^2/2)$.

## Why it matters
Rademacher variables are essential for defining Rademacher complexity, a key tool for bounding estimation error in infinite classes.

## Active recall
> [!question]- What is the MGF of a Rademacher random variable?
> $\mathbb{E}(e^{\alpha \varepsilon}) = \frac{1}{2}(e^\alpha + e^{-\alpha}) = \cosh(\alpha)$.

## When to use
When you need to prove concentration for sums of symmetric discrete variables.

## Core pattern
Verify the sub-Gaussian property by term-by-term comparison of the MGF's Taylor series with the Gaussian MGF series.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Prerequisites:**
- [[obj_036 - Definition 2]]
