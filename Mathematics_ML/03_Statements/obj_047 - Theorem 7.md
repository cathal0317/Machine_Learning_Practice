---
id: obj_047
title: "Theorem 7"
types:
  - theorem
page_start: 23
page_end: 23
parent_id: "obj_048"
children_ids:
  - obj_046
sibling_ids:
  - obj_049
  - obj_050
prerequisites:
  - obj_044
used_in:
  - obj_054
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 7

## Conceptual overview
This theorem provides a concrete mathematical guarantee for finite models. It states that the distance between the risk of our empirical minimizer and the best possible risk in our class decreases as we get more samples, but increases with the size of our search space (hypothesis class).

## Why it matters
It is the first major result in the module that provides a non-asymptotic bound on 'how well we have learned', accounting for both the data size and the model complexity.

## Active recall
> [!question]- What is the exact dependence of the estimation error on the class size $|\mathcal{H}|$ in Theorem 7?
> The error depends on $\sqrt{\log(|\mathcal{H}|)}$.

> [!question]- How does the range $L$ of the loss function affect the error bound?
> The bound is directly proportional to $L$, meaning higher variance in loss values leads to looser guarantees.

## Exact statement
Assume that the loss function satisfies $\ell : \mathcal{Y} \times \mathcal{Y} \to (0, L)$. Suppose $\mathcal{H}$ is finite and $\bar{h} = \text{argmin}_{h \in \mathcal{H}} R(h)$. Then, with probability at least $1 - \delta$, the ERM $\hat{h}$ satisfies 
$$
R(\hat{h}) - R(\bar{h}) \le L \sqrt{\frac{2 \log(|\mathcal{H}|) + 2 \log(\delta^{-1})}{n}}.
$$

## Links
**Parent:** [[obj_048 - 1.6.2 Finite hypothesis classes]]

**Children:**
- [[obj_046 - Proof of Theorem 7]]

**Prerequisites:**
- [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]

**Used in:**
- [[obj_054 - Remark 8]]
