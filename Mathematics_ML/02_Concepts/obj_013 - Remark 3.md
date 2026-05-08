---
id: obj_013
title: "Remark 3"
types:
  - remark
page_start: 7
page_end: 7
parent_id: "obj_014"
children_ids: []
sibling_ids:
  - obj_012
  - obj_016
  - obj_017
prerequisites:
  - obj_011
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 3

## Conceptual overview
This remark links the Bayes classifier to broader statistical frameworks. It notes that the regression function $\eta(x)$ is identical to the conditional expectation $E(Y | X = x)$ for binary labels. Furthermore, the decision rule is equivalent to a 'Maximum A Posteriori' (MAP) estimator, which picks the label that maximizes the posterior probability $P(Y = y | X = x)$.

## Why it matters
It establishes a dictionary between statistical learning theory, Bayesian inference, and regression analysis.

## Active recall
> [!question]- Why is the Bayes classifier called a 'maximum a posteriori estimator'?
> Because it chooses the label $y$ that maximizes the posterior probability $\mathbb{P}(Y = y \mid X = x)$.

## Mental picture
Imagine two overlapping probability density curves for class 0 and class 1; the MAP estimator simply tells you to pick whichever curve is higher at your current point $x$.

## Common confusions
Confusing the posterior probability $P(Y|X)$ with the likelihood $P(X|Y)$.

## Links
**Parent:** [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]

**Prerequisites:**
- [[obj_011 - 1.2 The Bayes classifier]]
