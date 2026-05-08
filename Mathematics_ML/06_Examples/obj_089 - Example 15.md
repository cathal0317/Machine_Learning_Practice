---
id: obj_089
title: "Example 15"
types:
  - example
page_start: 37
page_end: 37
parent_id: "obj_086"
children_ids: []
sibling_ids:
  - obj_084
  - obj_087
  - obj_088
  - obj_090
  - obj_092
  - obj_094
  - obj_095
  - obj_096
prerequisites:
  - obj_088
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 15

## Conceptual overview
This example considers the minimization of $\exp(-x)$ subject to $h(x, y) \le 0$, where $h(x, y) = x^2/y$ defined on $y > 0$. It shows that while the primal minimum is 1, the dual maximum is 0, illustrating a duality gap despite the problem being convex.

## Why it matters
It serves as a counterexample to the intuition that convexity alone is enough to guarantee strong duality.

## Active recall
> [!question]- Why does strong duality fail in Example 15?
> Strong duality fails because there is no 'gap' in the feasible region that allows the dual lower bound to reach the primal minimum; specifically, the problem lacks a point in the interior of the domain that satisfies the constraints strictly.

## When to use
Use this example to illustrate that even well-behaved convex problems can have a duality gap.

## Core pattern
Explicitly calculate both the primal minimum and the dual maximum to check if they are equal.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Prerequisites:**
- [[obj_088 - Definition 9 (Weak and strong duality)]]
