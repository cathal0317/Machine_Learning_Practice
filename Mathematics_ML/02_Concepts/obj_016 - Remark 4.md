---
id: obj_016
title: "Remark 4"
types:
  - remark
page_start: 8
page_end: 8
parent_id: "obj_014"
children_ids: []
sibling_ids:
  - obj_012
  - obj_013
  - obj_017
prerequisites:
  - obj_012
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 4

## Conceptual overview
This remark observes that the risk of the Bayes classifier $R(h^*)$ is equal to $E(\min(\eta(X), 1 - \eta(X)))$. Crucially, this value is always less than or equal to 0.5. It also provides the specific result for independent $X, Y$, where the risk simplifies to $\min(P(Y=1), P(Y=0))$.

## Why it matters
It defines the 'Bayes error', which is the irreducible noise in the dataset. Even the best possible model cannot perform better than this.

## Active recall
> [!question]- What is the integral expression for the Bayes risk $R(h^*)$?
> $R(h^*) = \mathbb{E}(\min(\eta(X), 1 - \eta(X)))$.

## Mental picture
Imagine you are trying to guess if a coin is weighted or fair. Even if you know exactly how it is weighted, if the coin lands heads 60% of the time, you will still be wrong 40% of the time. That 40% is the Bayes risk.

## Common confusions
Thinking the Bayes risk is always zero (it is only zero if labels are deterministic functions of features).

## Links
**Parent:** [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]

**Prerequisites:**
- [[obj_012 - Proposition 1]]
