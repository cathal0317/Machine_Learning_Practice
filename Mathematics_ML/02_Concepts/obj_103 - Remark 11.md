---
id: obj_103
title: "Remark 11"
types:
  - remark
page_start: 43
page_end: 43
parent_id: "obj_104"
children_ids: []
sibling_ids:
  - obj_105
  - obj_106
prerequisites:
  - obj_104
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 11

## Conceptual overview
While the feature map $\Phi(x)$ creates a nonlinear decision boundary in the input space $X$, the prediction function $h(x) = w^\top \Phi(x) + b$ is still a linear function of the parameters $w$ and $b$.

## Why it matters
This justifies the continued use of linear optimization theory to solve problems that result in nonlinear classifiers.

## Active recall
> [!question]- Is a kernel SVM a linear or nonlinear model?
> It is nonlinear in the input features but remains a linear model with respect to its parameters $w$ and $b$.

## Mental picture
Think of a 'warped' space where the decision boundary is straight, but when you look back at the original space, that straight line appears as a curve.

## Common confusions
Students often mistake 'nonlinear classifier' for 'nonlinear optimization problem', but for SVMs, the optimization remains a convex quadratic program regardless of the kernel.

## Links
**Parent:** [[obj_104 - 2.3.5 Nonlinear decision boundaries]]

**Prerequisites:**
- [[obj_104 - 2.3.5 Nonlinear decision boundaries]]
