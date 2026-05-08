---
id: obj_030
title: "A naive approach"
types:
  - discussion
page_start: 19
page_end: 19
parent_id: "obj_031"
children_ids: []
sibling_ids:
  - obj_035
  - obj_048
  - obj_051
  - obj_061
prerequisites:
  - obj_018
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# A naive approach

## Conceptual overview
For finite hypothesis classes, a naive approach to bounding estimation error uses the union bound over each function in the class. It demonstrates that as the number of samples increases, the probability of the empirical risk deviating significantly from the true risk for any function in the class decreases.

## Why it matters
It serves as a gateway to the more advanced concentration inequalities required for infinite hypothesis classes.

## Active recall
> [!question]- Why is the Central Limit Theorem insufficient for bounding estimation error in this context?
> The CLT is an asymptotic result (as $n \to \infty$) and does not provide quantitative bounds for finite $n$ that hold uniformly over a hypothesis class.

## Narrative flow
It defines the estimation error bound in terms of a supremum over the hypothesis class and uses a union bound over a finite set of random variables to guess the necessary sample complexity.

## Links
**Parent:** [[obj_031 - 1.6 The estimation error]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
