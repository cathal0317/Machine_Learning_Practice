---
id: obj_006
title: "1 Statistical learning"
types:
  - section
page_start: 5
page_end: 30
parent_id: "mod_01"
children_ids:
  - obj_005
  - obj_011
  - obj_018
  - obj_022
  - obj_028
  - obj_031
sibling_ids:
  - obj_001
  - obj_002
  - obj_069
  - obj_134
prerequisites:
  - obj_005
used_in:
  - obj_069
  - obj_134
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# 1 Statistical learning

## Conceptual overview
This chapter builds the theoretical foundation for machine learning by defining risk as the expected loss over an unknown probability distribution. It explores the ideal 'Bayes classifier' as the theoretical limit of performance and then shifts to the practical reality of empirical risk minimization (ERM) on finite datasets. The chapter concludes with rigorous bounds on estimation error using tools from concentration of measure like Rademacher complexity and VC dimension.

## Why it matters
It provides the 'why' behind machine learning algorithms, allowing practitioners to reason about how much data they need and what kind of model complexity is appropriate to avoid overfitting.

## Active recall
> [!question]- What is the central problem addressed in the Statistical Learning chapter?
> The central problem is choosing a hypothesis $h$ that minimizes the generalization risk $R(h)$ when the underlying joint distribution $P_0$ is unknown.

## Narrative flow
The section flows from definitions of types of learning and loss to the derivation of the optimal Bayes classifier. It then introduces the reality of having only data (ERM), analyzes the trade-offs involved (bias-variance), and develops advanced tools to bound the error of these practical estimators.

## Links
**Parent:** [[mod_01 - MA3K1 - Mathematics of machine learning]]

**Children:**
- [[obj_005 - 1.1 Classification and regression]]
- [[obj_011 - 1.2 The Bayes classifier]]
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
- [[obj_022 - 1.4 Bias-variance trade-off]]
- [[obj_028 - 1.5 Excess risk]]
- [[obj_031 - 1.6 The estimation error]]

**Prerequisites:**
- [[obj_005 - 1.1 Classification and regression]]

**Used in:**
- [[obj_069 - 2 Optimization]]
- [[obj_134 - Chapter 3 Neural networks]]
