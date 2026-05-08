---
id: obj_044
title: "Corollary 6 (Hoeffding’s inequality)"
types:
  - corollary
page_start: 23
page_end: 23
parent_id: "obj_035"
children_ids:
  - obj_045
sibling_ids:
  - obj_032
  - obj_033
  - obj_034
  - obj_036
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_043
prerequisites:
  - obj_040
  - obj_043
used_in:
  - obj_047
  - obj_059
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Corollary 6 (Hoeffding’s inequality)

## Conceptual overview
Hoeffding's inequality provides a sharp bound on the probability that the sum of independent bounded random variables deviates from its expected value. It shows that the tail probability decays exponentially with the square of the deviation, similar to a Gaussian distribution.

## Why it matters
It is a central tool for proving that empirical averages converge to their true expectations at a fast rate, which is the basis for statistical learning theory.

## Active recall
> [!question]- What are the requirements for using Hoeffding's inequality?
> The random variables must be independent and bounded within known intervals $(a_i, b_i)$.

> [!question]- How does the bound in Hoeffding's inequality change with the number of samples $n$?
> The tail probability decreases exponentially as $n$ increases, since $n$ appears in the negative exponent.

## Exact statement
If $W_1, \dots, W_n$ are independent and satisfy $a_i \le W_i \le b_i$ for all $i$. Then, for all $t \ge 0$, $Z := \frac{1}{n} \sum_{i=1}^n W_i$ satisfies 
$$
\mathbb{P}(Z - \mathbb{E}(Z) \ge t) \le \exp\left( -\frac{2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2} \right)
$$
 and 
$$
\mathbb{P}(|Z - \mathbb{E}(Z)| \ge t) \le 2 \exp\left( -\frac{2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2} \right).
$$

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Children:**
- [[obj_045 - Proof of Corollary 6]]

**Prerequisites:**
- [[obj_040 - Lemma 4 (Hoeffding’s lemma)]]
- [[obj_043 - Proposition 5]]

**Used in:**
- [[obj_047 - Theorem 7]]
- [[obj_059 - Proof of Lemma 10]]
