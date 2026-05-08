---
id: obj_020
title: "Remark 5"
types:
  - remark
page_start: 10
page_end: 10
parent_id: "obj_018"
children_ids: []
sibling_ids:
  - obj_019
  - obj_021
prerequisites:
  - obj_018
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 5

## Conceptual overview
This remark clarifies the statistical relationship between empirical risk and true risk. It states that for any fixed hypothesis $h$, the empirical risk $\hat{R}(h)$ is an unbiased estimator of the true risk $R(h)$, provided the training samples are independent and identically distributed (i.i.d.).

## Why it matters
It justifies the ERM approach by showing that, on average, the training error reflects the true error.

## Active recall
> [!question]- What is the expectation of the empirical risk $\hat{R}(h)$ for a fixed $h$?
> $\mathbb{E}(\hat{R}(h)) = R(h)$, assuming the data are i.i.d.

## Mental picture
Think of empirical risk as a single noisy measurement of the true risk; if you could repeat the whole experiment many times, the average of these measurements would perfectly match the true risk.

## Common confusions
Students often assume this unbiasedness also applies to $\hat{R}(\hat{h})$, where $\hat{h}$ is the minimizer. In fact, training error is typically a biased (underestimated) version of true error because the model was chosen specifically to minimize it.

## Links
**Parent:** [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
