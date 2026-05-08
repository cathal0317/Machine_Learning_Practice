---
id: obj_034
title: "The Chernoff bound"
types:
  - concept
page_start: 20
page_end: 20
parent_id: "obj_035"
children_ids: []
sibling_ids:
  - obj_032
  - obj_033
  - obj_036
  - obj_037
  - obj_038
  - obj_039
  - obj_040
  - obj_043
  - obj_044
prerequisites:
  - obj_033
used_in:
  - obj_035
  - obj_038
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# The Chernoff bound

## Conceptual overview
The Chernoff bound is an exponential tail bound obtained by applying Markov's inequality to the random variable $e^{\alpha W}$. It allows for much tighter (exponential) bounds than Markov's by optimizing over the parameter $\alpha$.

## Why it matters
It is the standard method for converting MGF information into tail probability bounds, leading to the sub-Gaussian theory used in machine learning.

## Active recall
> [!question]- What is the general form of the Chernoff bound for a random variable $W$?
> $\mathbb{P}(W \ge t) \le \inf_{\alpha > 0} e^{-\alpha t} \mathbb{E}(e^{\alpha W})$.

## Mental picture
Applying an exponential filter to a random variable to 'amplify' its tail behavior and then bounding it with a simple mean-based estimate.

## Common confusions
Forgetting that the parameter $\alpha$ must be positive and must be optimized to achieve the tightest possible bound.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Prerequisites:**
- [[obj_033 - Markov's inequality]]

**Used in:**
- [[obj_035 - 1.6.1 Tools from probability]]
- [[obj_038 - Proposition 3]]
