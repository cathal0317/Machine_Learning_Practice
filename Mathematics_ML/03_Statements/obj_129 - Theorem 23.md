---
id: obj_129
title: "Theorem 23"
types:
  - theorem
page_start: 53
page_end: 53
parent_id: "obj_127"
children_ids:
  - obj_130
sibling_ids:
  - obj_124
  - obj_125
  - obj_126
  - obj_128
  - obj_131
  - obj_132
prerequisites:
  - obj_126
  - obj_116
used_in:
  - obj_132
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 23

## Conceptual overview
Theorem 23 provides a formal convergence bound for SGD under the assumptions of strong convexity and bounded gradient variance. It establishes that the expected squared distance to the optimum decays at an $O(1/k)$ rate when using an appropriate stepsize schedule.

## Why it matters
It is the foundational theoretical guarantee for the convergence of stochastic optimization, giving confidence that SGD will eventually reach the optimum.

## Active recall
> [!question]- What is the expected convergence rate of SGD for strongly convex functions according to Theorem 23?
> The rate is $\mathbb{E}(\|w_k - w^*\|^2) \le R/(k+1)$, which is a sublinear $O(1/k)$ rate.

> [!question]- What additional assumption about the gradients is required for Theorem 23?
> It is assumed that the norm of the gradients is bounded, i.e., $\|\nabla f_i(w)\|^2 \le C^2$ for all $i$.

## Exact statement
Assume that $f$ is $\mu$-strongly convex and $\|\nabla f_i(w)\|^2 \le C^2$ for all $i$. Let $\tau_k = 1/(\mu(k+1))$. Then, $\mathbb{E}\|w_k - w^*\|^2 \le R/(k+1)$, where $R = \max(\|w_0 - w^*\|^2, C^2/\mu^2)$.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Children:**
- [[obj_130 - Proof of Theorem 23]]

**Prerequisites:**
- [[obj_126 - Stochastic gradient descent (SGD) Algorithm]]
- [[obj_116 - Definition 10 (Strong convexity and smoothness)]]

**Used in:**
- [[obj_132 - Remark 17]]
