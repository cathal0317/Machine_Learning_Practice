---
id: obj_039
title: "Remark 6"
types:
  - remark
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
  - obj_038
  - obj_040
  - obj_043
  - obj_044
prerequisites:
  - obj_036
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 6

## Conceptual overview
These remarks detail closure properties of sub-Gaussian variables. Specifically, if $W$ is sub-Gaussian, then $-W$ and $W - c$ are also sub-Gaussian with the same parameter.

## Why it matters
Understanding these properties simplifies the handling of centered and negated variables in estimation error proofs.

## Active recall
> [!question]- If $W$ is sub-Gaussian with parameter $\sigma$, is it also sub-Gaussian with parameter $\sigma' > \sigma$?
> Yes, the sub-Gaussian property is monotonic in the parameter.

## Mental picture
A set of simple rules allowing you to manipulate random variables (shifting, flipping) without losing the 'Gaussian-like' tail behavior.

## Common confusions
Confusing the sub-Gaussian parameter with the variance; they are related but not identical, though for a Gaussian $\mathcal{N}(0, \sigma^2)$, they coincide.

## Links
**Parent:** [[obj_035 - 1.6.1 Tools from probability]]

**Prerequisites:**
- [[obj_036 - Definition 2]]
