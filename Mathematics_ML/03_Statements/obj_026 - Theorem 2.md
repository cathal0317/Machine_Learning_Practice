---
id: obj_026
title: "Theorem 2"
types:
  - theorem
page_start: 16
page_end: 16
parent_id: "obj_028"
children_ids: []
sibling_ids:
  - obj_025
  - obj_027
  - obj_029
prerequisites:
  - obj_011
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 2

## Conceptual overview
The 'No Free Lunch' theorem proves that for any learning algorithm, there exists a distribution where it performs poorly (specifically, no better than random guessing). This implies that successful learning requires making prior assumptions about the hypothesis class or the data distribution.

## Why it matters
It justifies the necessity of inductive bias—restricting the hypothesis class—because learning cannot be universal and assumption-free.

## Active recall
> [!question]- What is the philosophical implication of the No Free Lunch theorem?
> It implies that learning is impossible without assumptions; you cannot have a universal algorithm that learns optimally for every possible distribution.

## Exact statement
Assume $\mathcal{X}$ is infinite. Fix $n \in \mathbb{N}$ and let $\epsilon > 0$. Consider a binary classification problem with unit loss and $\mathcal{Y} = \{0, 1\}$. Suppose we have an algorithm that outputs a predictor function $\hat{h}_n$, given from data $D_n = \{(X_i, Y_i)\}_{i=1}^n$. Then, there exists a probability distribution $P_0$ on $\mathcal{X} \times \{0, 1\}$ such that $\mathbb{E}_{D_n \sim P_0^n} \mathcal{E}(\hat{h}_n) \ge 1/2 - \epsilon$.

## Links
**Parent:** [[obj_028 - 1.5 Excess risk]]

**Prerequisites:**
- [[obj_011 - 1.2 The Bayes classifier]]
