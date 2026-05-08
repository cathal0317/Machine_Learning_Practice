---
id: obj_070
title: "Definition 5"
types:
  - definition
page_start: 32
page_end: 32
parent_id: "obj_068"
children_ids: []
sibling_ids:
  - obj_071
prerequisites: []
used_in:
  - obj_071
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 5

## Conceptual overview
A global minimizer is the absolute lowest point of a function. A local minimizer is the lowest point within a specific neighborhood. Strict minimizers are uniquely the lowest in their vicinity, while isolated minimizers are the only local optima in a neighborhood.

## Why it matters
In many non-convex problems (like deep learning), we often settle for local minimizers because finding global ones is intractable.

## Active recall
> [!question]- Can a function have a local minimizer but no global minimizer?
> Yes, for example $f(w) = -w^2$ has no global minimizer as it goes to $-\infty$, or $f(w) = \exp(-w)$ which has an infimum but no attainable global minimizer.

## Mental picture
Imagine a mountain range. The lowest point in the entire range is the global minimum. The bottom of any specific valley is a local minimum. If the valley floor is a single point, it is strict.

## Common confusions
Confusing a function that has an infimum (like $e^{-x}$) with one that has an attainable minimizer.

## Links
**Parent:** [[obj_068 - 2.1 Preliminaries]]

**Used in:**
- [[obj_071 - 2.1.1 Convexity]]
