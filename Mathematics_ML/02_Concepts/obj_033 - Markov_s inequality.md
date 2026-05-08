---
id: obj_033
title: "Markov's inequality"
types:
  - concept
page_start: 20
page_end: 20
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_032
  - obj_034
  - obj_036
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites: []
used_in:
  - obj_034
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Markov's inequality

## Conceptual overview
Markov's inequality provides a basic tail bound for any non-negative random variable $W$, stating $\mathbb{P}(W \ge t) \le \mathbb{E}(W)/t$. It relies on the simple observation that the probability of a variable exceeding a threshold is limited by its expectation.

## Why it matters
It is the foundational inequality from which Chernoff bounds and almost all more sophisticated concentration results are derived.

## Active recall
> [!question]- State Markov's inequality for a non-negative random variable $W$.
> $\mathbb{P}(W \ge t) \le \frac{\mathbb{E}(W)}{t}$ for all $t > 0$.

## Mental picture
Think of income: if the average income is $50k, at most 10% of people can earn more than$500k; otherwise, those 10% would already push the average higher than $50k.

## Common confusions
Applying Markov's inequality to a random variable that can take negative values without taking an absolute value first.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Used in:**
- [[obj_034 - The Chernoff bound]]
