---
id: obj_036
title: "Definition 2"
types:
  - definition
page_start: 21
page_end: 21
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_032
  - obj_033
  - obj_034
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites: []
used_in:
  - obj_038
  - obj_040
  - obj_043
  - obj_059
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 2

## Conceptual overview
A random variable $W$ is sub-Gaussian with parameter $\sigma > 0$ if its centered MGF is bounded by $\exp(\alpha^2\sigma^2/2)$ for all $\alpha$. This property implies that the distribution's tails decay at least as quickly as those of a Gaussian distribution with variance $\sigma^2$.

## Why it matters
The sub-Gaussian property allows us to apply Chernoff-style exponential bounds to a wide class of distributions beyond the Normal distribution.

## Active recall
> [!question]- State the MGF condition for a random variable $W$ to be sub-Gaussian with parameter $\sigma$.
> $\mathbb{E}(e^{\alpha(W-\mathbb{E}(W))}) \le e^{\alpha^2\sigma^2/2}$ for all $\alpha \in \mathbb{R}$.

## Mental picture
Imagine a Gaussian bell curve acting as a protective 'roof' for the distribution's tails; as long as the distribution stays below this Gaussian roof, it is sub-Gaussian.

## Common confusions
Assuming all distributions with finite variance are sub-Gaussian; sub-Gaussianity is much stronger as it requires exponential decay of the tails.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Used in:**
- [[obj_038 - Proposition 3]]
- [[obj_040 - Lemma 4 (Hoeffding’s lemma)]]
- [[obj_043 - Proposition 5]]
- [[obj_059 - Proof of Lemma 10]]
