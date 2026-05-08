---
id: obj_106
title: "Practical considerations"
types:
  - discussion
page_start: 45
page_end: 45
parent_id: "obj_104"
children_ids: []
sibling_ids:
  - obj_103
  - obj_105
prerequisites:
  - obj_105
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Practical considerations

## Conceptual overview
Choosing a kernel involves a bias-variance trade-off; higher-dimensional feature maps (like those with small $\sigma$ or high $d$) reduce approximation error but increase estimation error. Kernel methods generally have $O(n^2)$ complexity because of the kernel matrix, making them best suited for small to medium datasets.

## Why it matters
Knowing the limitations and tuning parameters (like $\sigma$ or degree) is essential for successfully applying SVMs in practice.

## Active recall
> [!question]- How should one choose the parameters $d$ or $\sigma$ for SVM kernels?
> These parameters should generally be selected using cross-validation to balance the trade-off between model fit and generalization error.

## Narrative flow
This subsection concludes the SVM discussion by focusing on parameter selection (hyperparameter tuning), the bias-variance trade-off, and computational complexity $O(n^2)$.

## Links
**Parent:** [[obj_104 - 2.3.5 Nonlinear decision boundaries]]

**Prerequisites:**
- [[obj_105 - Kernels]]
