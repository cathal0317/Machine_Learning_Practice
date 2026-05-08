---
id: obj_105
title: "Kernels"
types:
  - discussion
page_start: 44
page_end: 45
parent_id: "obj_104"
children_ids:
  - obj_107
sibling_ids:
  - obj_103
  - obj_106
prerequisites:
  - obj_104
used_in:
  - obj_106
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Kernels

## Conceptual overview
A kernel $k(x, x')$ is a function that computes the inner product $\Phi(x)^\top \Phi(x')$ without needing to explicitly calculate the high-dimensional vectors $\Phi(x)$. Common kernels include polynomial, Gaussian (Radial Basis Function), and sigmoid kernels.

## Why it matters
Kernels allow SVMs to work in infinite-dimensional spaces with the computational cost of the original input space, drastically increasing the model's capacity.

## Active recall
> [!question]- Define the Gaussian (RBF) kernel.
> $k(x, x') = \exp(-\|x - x'\|^2 / (2\sigma^2))$.

> [!question]- Why is the kernel formulation efficient?
> Because the dual SVM problem only depends on inner products, replacing them with a kernel allows us to skip the expensive or impossible step of computing high-dimensional feature vectors.

## Narrative flow
The discussion defines kernels, introduces standard specific kernel functions (Polynomial, Gaussian), and explains how the decision function is expressed in terms of kernels.

## Links
**Parent:** [[obj_104 - 2.3.5 Nonlinear decision boundaries]]

**Children:**
- [[obj_107 - Remark 12]]

**Prerequisites:**
- [[obj_104 - 2.3.5 Nonlinear decision boundaries]]

**Used in:**
- [[obj_106 - Practical considerations]]
