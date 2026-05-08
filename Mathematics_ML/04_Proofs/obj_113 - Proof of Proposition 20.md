---
id: obj_113
title: "Proof of Proposition 20"
types:
  - proof
page_start: 48
page_end: 48
parent_id: "obj_114"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_114
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 20

## Conceptual overview
The proof demonstrates that the error vector at each step of gradient descent on a quadratic objective is contracted by an operator related to the Hessian. By bounding the norm of this contraction operator using the Hessian's eigenvalues, a linear decay in the distance to the optimum is established.

## Why it matters
This proof explicitly connects the algebraic properties of the objective's Hessian to the practical performance of the gradient descent algorithm.

## Active recall
> [!question]- How is the error update represented in the proof of Proposition 20?
> The error at step $k+1$ is related to step $k$ via the operator $(Id - \tau_k C)$, such that $w_{k+1} - w^* = (Id - \tau_k C)(w_k - w^*)$.

> [!question]- How does the stepsize $\tau_k$ affect the contraction factor in this proof?
> The contraction factor $\rho$ is determined by $\|Id - \tau_k C\|$. Choosing $\tau_k$ such that this norm is strictly less than 1 ensures convergence.

## Proves
[[obj_114 - Proposition 20]]

## Proof skeleton
**Goal.** Show that the sequence $w_k$ converges to $w^*$ linearly with respect to the error norm.

**Strategy.** Establish a recursive relation for the error vector $w_k - w^*$ and bound the norm of the transition matrix using eigenvalues.

- Write the gradient update $w_{k+1} = w_k - \tau_k(Cw_k - b)$ and note that $Cw^* = b$.

- Subtract $w^*$ from both sides to obtain $w_{k+1} - w^* = (Id - \tau_k C)(w_k - w^*)$.

- Use the property $\|w_{k+1} - w^*\| \le \|Id - \tau_k C\| \cdot \|w_k - w^*\|$.

- Evaluate the operator norm $\|Id - \tau_k C\|$ as $\max(|1 - \tau_k L|, |1 - \tau_k \mu|)$.

- Show that for $\tau_k < 2/L$, this maximum is strictly less than 1.

**Conclusion.** By induction, $\|w_k - w^*\| \le \rho^k \|w_0 - w^*\|$.

## Links
**Parent:** [[obj_114 - Proposition 20]]

**Prerequisites:**
- [[obj_114 - Proposition 20]]
