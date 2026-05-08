---
id: obj_021
title: "Example 3"
types:
  - example
page_start: 10
page_end: 11
parent_id: "obj_018"
children_ids: []
sibling_ids:
  - obj_019
  - obj_020
prerequisites:
  - obj_018
used_in:
  - obj_023
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 3

## Conceptual overview
K-nearest neighbors (K-NN) is a local, non-parametric classification method that predicts the label of a new point based on a 'vote' from its closest neighbors in the training set. Unlike linear regression which uses fixed parameters, K-NN complexity grows with the data as it implicitly partitions the input space into Voronoi regions.

## Why it matters
It represents a fundamental approach to classification that does not assume a specific functional form for the decision boundary, making it highly flexible but sensitive to the choice of the hyperparameter $K$.

## Active recall
> [!question]- What is the difference between 1-NN and K-NN in terms of noise sensitivity?
> 1-NN is highly sensitive to noise or mislabeled examples because its prediction depends on a single training point; K-NN reduces this sensitivity by averaging over $K$ points.

> [!question]- How is the hypothesis class $\mathcal{H}$ defined for a 1-NN classifier?
> It consists of indicator functions on Voronoi cells, where each cell contains points closer to a specific training example than to any other.

## When to use
When you have a classification task where local similarity is a strong indicator of class membership and the decision boundary is potentially complex.

## Core pattern
Identify the $K$ nearest neighbors of an input point using a distance metric and assign the class label that is most frequent among those neighbors.

## Links
**Parent:** [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]

**Used in:**
- [[obj_023 - 1.4.1 K nearest neighbours]]
