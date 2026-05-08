---
id: obj_052
title: "Definition 3 (Rademacher complexity)"
types:
  - definition
page_start: 25
page_end: 25
parent_id: "obj_051"
children_ids: []
sibling_ids:
  - obj_053
  - obj_054
  - obj_055
  - obj_056
  - obj_057
prerequisites: []
used_in:
  - obj_055
  - obj_056
  - obj_058
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 3 (Rademacher complexity)

## Conceptual overview
The empirical version is calculated for a fixed dataset, while the population version is the expectation of the empirical version over all possible datasets. It involves taking a supremum over the function class of the inner product between function values and random labels.

## Why it matters
It allows us to talk about the 'size' of a set of functions $\mathcal{H}$ in a way that respects the data distribution.

## Active recall
> [!question]- What is the difference between empirical and population Rademacher complexity?
> Empirical complexity is fixed for a given dataset $z_{1:n}$, while population complexity is the expectation over the data $\mathcal{R}_n(\mathcal{F}) = \mathbb{E}(\hat{\mathcal{R}}(\mathcal{F}(Z_{1:n})))$.

## Mental picture
Imagine giving a model random coins (±1) as labels for its training data. Rademacher complexity asks: 'On average, how well can the best function in your model agree with these random coins?'

## Common confusions
Confusing the expectation over Rademacher variables $\varepsilon$ with the expectation over data $Z$. Population complexity involves both.

## Links
**Parent:** [[obj_051 - 1.6.3 Rademacher complexity]]

**Used in:**
- [[obj_055 - Theorem 8]]
- [[obj_056 - Theorem 9]]
- [[obj_058 - Lemma 10. [Massart’s Lemma]]]
