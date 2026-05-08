---
id: obj_064
title: "Remark 9"
types:
  - remark
page_start: 28
page_end: 28
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_062
  - obj_063
  - obj_065
  - obj_066
  - obj_067
prerequisites:
  - obj_062
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 9

## Conceptual overview
To prove $VC(\mathcal{H}) = n$, one must satisfy two conditions: first, identify at least one configuration of $n$ points that can be shattered; second, prove that no possible configuration of $n+1$ points can be shattered.

## Why it matters
It clarifies that VC dimension is defined by the existence of a 'good' set of points, rather than a universal property of all sets.

## Active recall
> [!question]- To show $VC(\mathcal{H}) \geq n$, do we need to show every set of size $n$ is shattered?
> No, we only need to find one set of $n$ distinct points that can be shattered.

## Mental picture
Think of it as an 'existence' check for the lower bound and a 'universal' check for the upper bound.

## Common confusions
Students often fail to check the upper bound part, correctly finding a set that is shattered but not proving that no larger set exists.

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_062 - Definition 4]]
