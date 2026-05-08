---
id: obj_107
title: "Remark 12"
types:
  - remark
page_start: 45
page_end: 45
parent_id: "obj_105"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_105
used_in:
  - obj_133
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 12

## Conceptual overview
When using a sigmoid kernel $K(x, x') = \tanh(\alpha(x^\top x') + \beta)$, the resulting SVM decision function $h(x) = \sum a_i K(X_i, x) + b$ structurally resembles a single-hidden-layer neural network where the 'weights' are the training data points.

## Why it matters
It provides a conceptual bridge between kernel-based methods and the neural network section that follows.

## Active recall
> [!question]- What is the connection between SVMs and Neural Networks?
> An SVM with a sigmoid kernel can be viewed as a simple neural network where the hidden layer neurons use the training data points as their internal weights.

## Mental picture
Imagine an SVM as a neural network that is 'lazy' and simply uses its favorite training examples as its internal logic nodes.

## Common confusions
While they look similar, SVMs and Neural Networks use very different optimization objectives (hinge loss vs. cross-entropy/MSE) and different training algorithms.

## Links
**Parent:** [[obj_105 - Kernels]]

**Prerequisites:**
- [[obj_105 - Kernels]]

**Used in:**
- [[obj_133 - 3.1 Multilayer perceptrons]]
