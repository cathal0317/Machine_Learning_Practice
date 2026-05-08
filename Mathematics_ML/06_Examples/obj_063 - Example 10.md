---
id: obj_063
title: "Example 10"
types:
  - example
page_start: 28
page_end: 28
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_062
  - obj_064
  - obj_065
  - obj_066
  - obj_067
prerequisites:
  - obj_062
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 10

## Conceptual overview
For the set of all possible classifiers on an infinite space, the VC dimension is infinite because any number of distinct points can be shattered. For a finite set of functions $\mathcal{H}$, the growth function is trivially bounded by $|\mathcal{H}|$, which implies the VC dimension is at most $\log_2(|\mathcal{H}|)$.

## Why it matters
It benchmarks the VC dimension against the previously studied finite case, showing it is a consistent generalization of complexity.

## Active recall
> [!question]- What is the VC dimension of a hypothesis class consisting of only two constant functions, $h_1 = 1$ and $h_2 = -1$?
> The VC dimension is 1, because it can shatter a single point (achieving both labels) but cannot shatter two points (which would require 4 labelings).

## When to use
Use these baseline cases to sanity-check VC dimension calculations for more complex classes.

## Core pattern
Bounding the VC dimension by the cardinality of the function set using $2^n \leq |\mathcal{H}|$.

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_062 - Definition 4]]
