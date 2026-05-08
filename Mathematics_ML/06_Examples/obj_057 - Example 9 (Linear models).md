---
id: obj_057
title: "Example 9 (Linear models)"
types:
  - example
page_start: 26
page_end: 27
parent_id: "obj_051"
children_ids: []
sibling_ids:
  - obj_052
  - obj_053
  - obj_054
  - obj_055
  - obj_056
prerequisites:
  - obj_052
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 9 (Linear models)

## Conceptual overview
This example performs a rigorous derivation for linear functions bounded by norm $K$. It demonstrates that the complexity scales with the expected norm of the feature vectors and inversely with the square root of the number of samples.

## Why it matters
Linear models are the bedrock of ML; this result proves their generalization ability is independent of the input dimensionality, depending instead on the norm of the features.

## Active recall
> [!question]- What is the Rademacher complexity $\mathcal{R}_n(\mathcal{H})$ for linear models $w^\top \phi(x)$ with $\|w\| \le K$?
> $\mathcal{R}_n(\mathcal{H}) \le \frac{K \sqrt{\mathbb{E}(\|\phi(X)\|^2)}}{\sqrt{n}}$.

## When to use
Use when analyzing the generalization of linear regression or support vector machines with feature maps.

## Core pattern
Use Jensen's inequality to move the expectation inside the square root, then use independence and the mean-zero property of Rademacher variables.

## Links
**Parent:** [[obj_051 - 1.6.3 Rademacher complexity]]

**Prerequisites:**
- [[obj_052 - Definition 3 (Rademacher complexity)]]
