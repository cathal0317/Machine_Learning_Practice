---
id: obj_038
title: "Proposition 3"
types:
  - proposition
page_start: 21
page_end: 21
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_032
  - obj_033
  - obj_034
  - obj_036
  - obj_037
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites:
  - obj_034
  - obj_036
used_in:
  - obj_060
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 3

## Conceptual overview
This proposition establishes that every sub-Gaussian random variable has a tail probability that decays at least exponentially fast: $\mathbb{P}(W - \mathbb{E}(W) \ge t) \le e^{-t^2/(2\sigma^2)}$. This confirms that the MGF definition translates into a concrete, useful probability bound.

## Why it matters
It is the main reason sub-Gaussian variables are studied; they guarantee that large deviations from the mean are extremely unlikely.

## Active recall
> [!question]- Given a sub-Gaussian variable $W$, what is the upper bound on $\mathbb{P}(|W - \mathbb{E}(W)| \ge t)$?
> $2 \exp(-t^2/(2\sigma^2))$.

## Exact statement
If $W$ is sub-Gaussian with parameter $\sigma > 0$, then $\mathbb{P}(W - \mathbb{E}(W) \ge t) \le e^{-t^2/(2\sigma^2)}$ for all $t \ge 0$.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Prerequisites:**
- [[obj_034 - The Chernoff bound]]
- [[obj_036 - Definition 2]]

**Used in:**
- [[obj_060 - Proposition 11]]
