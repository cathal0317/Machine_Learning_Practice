---
id: obj_081
title: "Proposition 16 (First order optimality condition - constrained)"
types:
  - proposition
page_start: 35
page_end: 35
parent_id: "obj_071"
children_ids:
  - obj_079
sibling_ids:
  - obj_072
  - obj_074
  - obj_076
  - obj_077
  - obj_080
  - obj_082
prerequisites:
  - obj_072
  - obj_071
used_in:
  - obj_082
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 16 (First order optimality condition - constrained)

## Conceptual overview
When optimization is restricted to a convex set $\mathcal{F}$, the optimum occurs when the gradient makes an obtuse or right angle with any feasible direction. This means you cannot move further into the set to decrease the function value.

## Why it matters
It explains why we find solutions on the boundary of feasible regions in constrained problems.

## Active recall
> [!question]- What is the condition for optimality in constrained convex optimization?
> For all $w \in \mathcal{F}$, we must have $\nabla f(w^*)^\top (w - w^*) \geq 0$.

## Exact statement
Let $f$ be convex and differentiable and consider $\min_{w \in \mathcal{F}} f(w)$ for some convex set $\mathcal{F}$. Then, $w^*$ is a minimizer if and only if $\forall w \in \mathcal{F}, \nabla f(w^*)^\top (w - w^*) \geq 0$.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Children:**
- [[obj_079 - Proof of Proposition 16]]

**Prerequisites:**
- [[obj_072 - Definition 6]]
- [[obj_071 - 2.1.1 Convexity]]

**Used in:**
- [[obj_082 - Remark 10]]
