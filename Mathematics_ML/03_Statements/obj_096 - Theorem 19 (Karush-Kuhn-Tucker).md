---
id: obj_096
title: "Theorem 19 (Karush-Kuhn-Tucker)"
types:
  - theorem
page_start: 38
page_end: 39
parent_id: "obj_086"
children_ids:
  - obj_097
sibling_ids:
  - obj_084
  - obj_087
  - obj_088
  - obj_089
  - obj_090
  - obj_092
  - obj_094
  - obj_095
prerequisites:
  - obj_094
  - obj_095
used_in:
  - obj_101
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 19 (Karush-Kuhn-Tucker)

## Conceptual overview
The KKT conditions are a set of first-order derivative requirements for a solution in nonlinear programming to be optimal. They involve stationarity (gradient of Lagrangian is zero), primal feasibility, dual feasibility, and complementary slackness.

## Why it matters
KKT conditions are the foundation for many optimization algorithms and provide a direct way to verify if a candidate solution is truly optimal.

## Active recall
> [!question]- Are KKT conditions sufficient for optimality?
> Yes, if the optimization problem is convex and strong duality holds, then any point satisfying the KKT conditions is a primal-dual optimal solution.

> [!question]- List the four main KKT conditions.
> 1. Stationary condition; 2. Primal feasibility; 3. Dual feasibility; 4. Complementary slackness.

## Exact statement
If $w^*$ and $(\xi^*, \nu^*)$ solve the primal and dual problems and strong duality holds, then: (1) Stationary condition: $\nabla f(w^*) + \sum_i \nu^*_i \nabla f_i(w^*) + A^\top \xi^* = 0$; (2) Primal feasibility: $f_i(w^*) \le 0$, $Aw^* = b$; (3) Dual feasibility: $\nu^* \ge 0$; (4) Complementary slackness: $\nu^*_j f_j(w^*) = 0$ for all $j = 1, \dots, m$.

## Links
**Parent:** [[obj_086 - 2.2.1 Duality]]

**Children:**
- [[obj_097 - Proof of Theorem 19]]

**Prerequisites:**
- [[obj_094 - The Karush-Kuhn-Tucker (KKT) conditions discussion]]
- [[obj_095 - Theorem 18 (Slater's constraint quantification)]]

**Used in:**
- [[obj_101 - 2.3.3 The dual problem for SVM]]
