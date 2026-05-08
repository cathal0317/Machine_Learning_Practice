---
id: obj_012
title: "Proposition 1"
types:
  - proposition
page_start: 7
page_end: 7
parent_id: "obj_014"
children_ids:
  - obj_015
sibling_ids:
  - obj_013
  - obj_016
  - obj_017
prerequisites:
  - obj_010
  - obj_011
used_in:
  - obj_015
  - obj_016
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 1

## Conceptual overview
This proposition characterizes the Bayes classifier in terms of the regression function $\eta(x) = P(Y = 1 | X = x)$. It states that the optimal decision rule is a simple threshold: predict class 1 if the conditional probability of class 1 exceeds 0.5, and predict class 0 otherwise. This result demonstrates that classification is fundamentally a probability estimation problem.

## Why it matters
It gives an explicit, constructive formula for the best possible classifier, reducing the search over all measurable functions to a simple thresholding of a conditional mean.

## Active recall
> [!question]- State the rule for the Bayes classifier $h^*(x)$ for binary labels $\{0, 1\}$.
> $h^*(x) = 1$ if $\eta(x) > 1/2$, and $h^*(x) = 0$ if $\eta(x) \leq 1/2$, where $\eta(x) = \mathbb{P}(Y = 1 \mid X = x)$.

## Exact statement
The Bayes classifier $h^*$ is given by 
$$
h^*(x) = \begin{cases} 1 & \eta(x) > \frac{1}{2} \\ 0 & \eta(x) \leq \frac{1}{2} \end{cases}
$$
 where $\eta(x) := \mathbb{P}(Y = 1 \mid X = x)$.

## Links
**Parent:** [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]

**Children:**
- [[obj_015 - Proof of Proposition 1]]

**Prerequisites:**
- [[obj_010 - 1.2.1 Conditional expectation]]
- [[obj_011 - 1.2 The Bayes classifier]]

**Used in:**
- [[obj_015 - Proof of Proposition 1]]
- [[obj_016 - Remark 4]]
