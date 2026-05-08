---
id: obj_019
title: "Example 2"
types:
  - example
page_start: 10
page_end: 10
parent_id: "obj_018"
children_ids: []
sibling_ids:
  - obj_020
  - obj_021
prerequisites:
  - obj_018
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 2

## Conceptual overview
This example demonstrates ERM in the context of linear regression using squared error loss. It shows that finding the weights $w$ and bias $b$ that minimize the training error is equivalent to solving a 'least squares' problem. The resulting system of equations leads to the classic normal equation solution using matrix algebra.

## Why it matters
It shows that for certain simple hypothesis classes and losses, the ERM can be computed exactly in closed form using linear algebra.

## Active recall
> [!question]- What is the closed-form solution for the parameters $(\hat{w}, \hat{b})$ in linear regression ERM?
> $\begin{pmatrix} \hat{w} \\ \hat{b} \end{pmatrix} = (M^\top M)^{-1} M^\top Y_{1:n}$, where $M$ is the feature matrix with an added column of ones.

## When to use
Use this when your hypothesis class consists of linear functions and you are minimizing the sum of squared residuals.

## Core pattern
Formulate the training error as a matrix-vector norm, take the gradient with respect to parameters, and set to zero to find the stationary point.

## Links
**Parent:** [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
