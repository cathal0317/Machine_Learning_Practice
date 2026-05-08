---
id: obj_055
title: "Theorem 8"
types:
  - theorem
page_start: 26
page_end: 26
parent_id: "obj_051"
children_ids: []
sibling_ids:
  - obj_052
  - obj_053
  - obj_054
  - obj_056
  - obj_057
prerequisites:
  - obj_052
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 8

## Conceptual overview
This theorem states that the average 'worst-case' deviation between true expectations and empirical averages across a set of functions is bounded by twice the population Rademacher complexity.

## Why it matters
It is the fundamental inequality that links the theoretical concept of Rademacher complexity to the practical problem of bounding deviations in empirical risk.

## Active recall
> [!question]- What is the mathematical statement of Theorem 8?
> $\mathbb{E}(\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n (\mathbb{E}f(Z_i) - f(Z_i))) \le 2\mathcal{R}_n(\mathcal{F})$.

## Exact statement
Let $\mathcal{F}$ be a class of real-valued functions $f : \mathcal{Z} \to \mathbb{R}$ and let $Z_1, \dots, Z_n$ be iid random variables taking values in $\mathcal{Z}$. Then, 
$$
\mathbb{E}\left( \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^n (\mathbb{E}f(Z_i) - f(Z_i)) \right) \le 2\mathcal{R}_n(\mathcal{F}).
$$

## Links
**Parent:** [[obj_051 - 1.6.3 Rademacher complexity]]

**Prerequisites:**
- [[obj_052 - Definition 3 (Rademacher complexity)]]
