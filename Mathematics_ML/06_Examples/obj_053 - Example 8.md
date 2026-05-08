---
id: obj_053
title: "Example 8"
types:
  - example
page_start: 25
page_end: 25
parent_id: "obj_051"
children_ids: []
sibling_ids:
  - obj_052
  - obj_054
  - obj_055
  - obj_056
  - obj_057
prerequisites:
  - obj_052
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 8

## Conceptual overview
It shows two extremes: a singleton set has zero complexity (it cannot fit noise at all), while the set of all possible binary classifiers has the maximum possible complexity of 1 (it can perfectly fit any noise).

## Why it matters
It establishes the scale and bounds of Rademacher complexity, showing it ranges from 0 to 1 for binary classification.

## Active recall
> [!question]- Why is the Rademacher complexity of a singleton set zero?
> Because for a fixed function $f$, the expectation $\mathbb{E}_{\varepsilon}(\sum \varepsilon_i f(z_i))$ sums to zero since $\mathbb{E}(\varepsilon_i) = 0$.

## When to use
Use these boundary cases to verify more complex derivations or to check if a complexity measure is normalized correctly.

## Core pattern
For singletons, use linearity of expectation. For the set of all classifiers, use the fact that the supremum can pick labels that match the noise.

## Links
**Parent:** [[obj_051 - 1.6.3 Rademacher complexity]]

**Prerequisites:**
- [[obj_052 - Definition 3 (Rademacher complexity)]]
