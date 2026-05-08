---
id: obj_114
title: "Proposition 20"
types:
  - proposition
page_start: 48
page_end: 48
parent_id: "obj_112"
children_ids:
  - obj_113
sibling_ids:
  - obj_115
prerequisites:
  - obj_108
used_in:
  - obj_115
analogous_to: []
same_pattern_as:
  - obj_121
family: "theorem-family"
---

# Proposition 20

## Conceptual overview
This proposition formalizes the linear convergence of gradient descent for quadratic objectives. It states that if the stepsize is chosen within a specific range related to the maximum eigenvalue of the Hessian, the iterates will converge to the unique minimizer at a rate governed by the Hessian's condition number.

## Why it matters
It provides a theoretical guarantee of convergence and a precise rate for one of the most fundamental classes of optimization problems in machine learning.

## Active recall
> [!question]- What is the range of stepsize $\tau_k$ that guarantees convergence in Proposition 20?
> The stepsize must satisfy $0 < \tau_{min} \le \tau_k \le \tau_{max} < 2/L$, where $L$ is the largest eigenvalue of the matrix $C$.

> [!question]- What is the optimal contraction constant $\tilde{\rho}$ according to Proposition 20?
> The optimal rate is $\tilde{\rho} = (L - \mu)/(L + \mu)$, achieved when $\tau_k = 2/(L + \mu)$.

## Exact statement
Let $f(w) = \frac{1}{2}\langle Cw, w \rangle - \langle w, b \rangle$ where $C$ is a symmetric positive definite matrix with eigenvalues in $(\mu, L)$ for $0 < \mu < L$. Assume that there exists $\tau_{min}, \tau_{max}$ such that $0 < \tau_{min} \le \tau_k \le \tau_{max} < 2/L$. Then, there exists $\rho \in (0, 1)$ such that $\|w_k - w^*\| \le \rho^k \|w_0 - w^*\|$.

## Links
**Parent:** [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]

**Children:**
- [[obj_113 - Proof of Proposition 20]]

**Prerequisites:**
- [[obj_108 - Gradient descent algorithm]]

**Used in:**
- [[obj_115 - Remark 13]]

**Same pattern as:**
- [[obj_121 - Theorem 22]]
