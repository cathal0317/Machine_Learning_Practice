---
id: obj_082
title: "Remark 10"
types:
  - remark
page_start: 35
page_end: 35
parent_id: "obj_071"
children_ids: []
sibling_ids:
  - obj_072
  - obj_074
  - obj_076
  - obj_077
  - obj_080
  - obj_081
prerequisites:
  - obj_080
  - obj_081
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 10

## Conceptual overview
If the feasible set is the entire space $\mathbb{R}^p$, the condition $\nabla f(w^*)^ op v \geq 0$ must hold for all directions $v$. Since this must hold for both $v$ and $-v$, it implies the gradient must be zero.

## Why it matters
It demonstrates the consistency and universality of the constrained optimality framework.

## Active recall
> [!question]- How does Proposition 16 reduce to Proposition 15 in the unconstrained case?
> In the unconstrained case, we can choose $w - w^*$ to be any vector $v$. If $\nabla f(w^*)^ op v \geq 0$ for all $v$, then it must be that $\nabla f(w^*) = 0$.

## Mental picture
Imagine a ball at the bottom of a bowl. In an open field, it sits where the floor is flat. If you place a wall in the field, it sits where it pushes against the wall and cannot roll further in that direction.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Prerequisites:**
- [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]
- [[obj_081 - Proposition 16 (First order optimality condition - constrained)]]
