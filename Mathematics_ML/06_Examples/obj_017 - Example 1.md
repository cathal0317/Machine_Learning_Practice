---
id: obj_017
title: "Example 1"
types:
  - example
page_start: 8
page_end: 9
parent_id: "obj_014"
children_ids: []
sibling_ids:
  - obj_012
  - obj_013
  - obj_016
prerequisites:
  - obj_014
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 1

## Conceptual overview
This example considers a mixture of two Gaussian distributions representing house prices in two different neighborhoods. It shows how to use Bayes' rule to calculate the regression function $\eta(x)$ from the class priors $P(Y=y)$ and class-conditional densities $f_{X|Y}$. The resulting Bayes classifier is shown to be a simple inequality comparing weighted densities.

## Why it matters
It bridges the gap between the abstract definition of the Bayes classifier and its concrete calculation in a standard statistical model.

## Active recall
> [!question]- In the Gaussian mixture model, how is $\eta(x)$ computed using class densities $\rho_0, \rho_1$?
> By Bayes' rule, $\eta(x) = \frac{q\rho_1(x)}{(1-q)\rho_0(x) + q\rho_1(x)}$, where $q = \mathbb{P}(Y=1)$.

## When to use
Use this approach when you have a generative model of the data, i.e., you know the distribution of the features within each class and the overall frequency of each class.

## Core pattern
Apply Bayes' rule to convert class-conditional densities and priors into a posterior probability, then apply the 0.5 threshold.

## Links
**Parent:** [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]

**Prerequisites:**
- [[obj_014 - 1.2.2 Characterization of the Bayes classifier]]
