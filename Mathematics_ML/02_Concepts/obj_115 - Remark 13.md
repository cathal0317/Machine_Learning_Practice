---
id: obj_115
title: "Remark 13"
types:
  - remark
page_start: 48
page_end: 48
parent_id: "obj_112"
children_ids: []
sibling_ids:
  - obj_114
prerequisites:
  - obj_114
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 13

## Conceptual overview
This remark introduces the inverse condition number $\kappa = \mu/L$ as the primary factor determining the difficulty of an optimization problem. It explains why 'ill-conditioned' problems, where $\kappa$ is small, lead to slow convergence for gradient descent.

## Why it matters
It provides intuition for why certain datasets or models are harder to optimize than others based on the 'stretched' nature of the objective's contours.

## Active recall
> [!question]- How is the inverse condition number $\kappa$ defined in Remark 13?
> It is defined as the ratio of the smallest to largest eigenvalues of the Hessian matrix, $\kappa = \mu / L$.

> [!question]- What happens to the convergence rate when the inverse condition number $\kappa$ is very small?
> When $\kappa$ is small, the contraction constant $\rho \approx 1 - 2\kappa$ becomes very close to 1, leading to very slow convergence.

## Mental picture
Imagine an objective function as a valley. A well-conditioned problem ($\kappa \approx 1$) is a circular bowl where any step toward the center is efficient. An ill-conditioned problem ($\kappa \ll 1$) is a long, thin 'trough' where the algorithm bounces between the steep side walls while making very slow progress along the flat floor.

## Common confusions
Confusing the condition number with the magnitude of the gradient; a problem can have a large condition number even if the gradients are small, and vice versa.

## Links
**Parent:** [[obj_112 - 2.4.1 Convergence analysis for gradient descent]]

**Prerequisites:**
- [[obj_114 - Proposition 20]]
