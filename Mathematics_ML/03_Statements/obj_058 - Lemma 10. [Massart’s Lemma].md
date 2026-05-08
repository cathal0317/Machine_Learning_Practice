---
id: obj_058
title: "Lemma 10. [Massart’s Lemma]"
types:
  - lemma
page_start: 27
page_end: 27
parent_id: "obj_061"
children_ids:
  - obj_059
sibling_ids:
  - obj_060
  - obj_062
  - obj_063
  - obj_064
  - obj_065
  - obj_066
  - obj_067
prerequisites:
  - obj_052
used_in:
  - obj_067
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Lemma 10. [Massart’s Lemma]

## Conceptual overview
Massart's Lemma bridges finite counting and Rademacher complexity. It states that if a function class restricted to a dataset has a finite number of 'behaviors' (vectors), then its Rademacher complexity is bounded by the log of that number of behaviors.

## Why it matters
It allows us to bound Rademacher complexity using combinatorial properties of the function class, leading directly to the VC dimension results.

## Active recall
> [!question]- What does Massart's Lemma relate?
> It relates empirical Rademacher complexity to the number of distinct behaviors $|\mathcal{F}(z_{1:n})|$ of a function class on a dataset.

## Exact statement
Let $\mathcal{F} := \{(x, y) \to \ell(h(x), y) ; h \in \mathcal{H}\}$ and assume that $\ell$ takes values in $(0, 1)$. Then, 
$$
\hat{\mathcal{R}}(\mathcal{F}(z_{1:n})) \le \sqrt{2 \log(|\mathcal{F}(z_{1:n})|)/n}.
$$

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Children:**
- [[obj_059 - Proof of Lemma 10]]

**Prerequisites:**
- [[obj_052 - Definition 3 (Rademacher complexity)]]

**Used in:**
- [[obj_067 - Lemma 12 (Sauer-Shelah Lemma)]]
