---
id: obj_121
title: "Theorem 22"
types:
  - theorem
page_start: 50
page_end: 50
parent_id: "obj_122"
children_ids:
  - obj_123
sibling_ids: []
prerequisites:
  - obj_117
  - obj_112
used_in: []
analogous_to: []
same_pattern_as:
  - obj_114
family: "theorem-family"
---

# Theorem 22

## Conceptual overview
Theorem 22 is a central result that specifies the convergence rates for gradient descent under different assumptions. For functions that are only smooth, the objective value converges at a sublinear rate of $O(1/k)$, whereas for functions that are both smooth and strongly convex, the iterates converge at a much faster linear rate.

## Why it matters
This theorem allows us to predict how many iterations are needed to reach a desired accuracy for a wide variety of machine learning models.

## Active recall
> [!question]- What is the convergence rate of gradient descent if a function is only Lipschitz smooth?
> The objective value $f(w_k)$ converges to the minimum $f(w^*)$ at a rate of $O(1/k)$.

> [!question]- Under what conditions does gradient descent achieve linear convergence?
> Linear convergence is achieved when the function is both $L$-Lipschitz smooth and $\mu$-strongly convex.

## Exact statement
Assume $f$ satisfies $(C_L)$. Let $0 < \tau_{min} \le \tau_k \le \tau_{max} < 2/L$. Then $\lim_{k \to \infty} w_k = w^*$ and $f(w_k) - f(w^*) \le C/k$ for some $C > 0$. If, in addition, $f$ is $\mu$-strongly convex, then there exists $\rho \in (0, 1)$ such that $\|w_k - w^*\| \le \rho^k \|w^* - w_0\|$.

## Links
**Parent:** [[obj_122 - 2.4.3 Gradient descent under strong convexity]]

**Children:**
- [[obj_123 - Proof of Theorem 22]]

**Prerequisites:**
- [[obj_117 - Proposition 21]]
- [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]

**Same pattern as:**
- [[obj_114 - Proposition 20]]
