---
id: obj_074
title: "Theorem 13"
types:
  - theorem
page_start: 33
page_end: 33
parent_id: "obj_071"
children_ids:
  - obj_073
sibling_ids:
  - obj_072
  - obj_076
  - obj_077
  - obj_080
  - obj_081
  - obj_082
prerequisites:
  - obj_072
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 13

## Conceptual overview
This theorem is the 'fundamental theorem' of convex optimization. It guarantees that any local optimum found by an algorithm is the best possible solution across the entire domain. It also establishes the uniqueness of minimizers under strict convexity.

## Why it matters
It explains why we don't need to worry about 'getting stuck' in bad local optima when the problem is convex.

## Active recall
> [!question]- What condition on a convex function ensures its minimizer is unique?
> Strict convexity ensures uniqueness.

## Exact statement
Let $f : \mathbb{R}^p \to \mathbb{R}$ be convex. Then any local minimizer of $f$ is a global minimizer. Moreover, if $f$ is strictly convex and has a minimizer, then it is unique.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Children:**
- [[obj_073 - Proof of Theorem 13]]

**Prerequisites:**
- [[obj_072 - Definition 6]]
