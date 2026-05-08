---
id: obj_049
title: "Example 7"
types:
  - example
page_start: 24
page_end: 24
parent_id: "obj_048"
children_ids: []
sibling_ids:
  - obj_047
  - obj_050
prerequisites:
  - obj_048
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 7

## Conceptual overview
This example considers a classifier defined on a grid of $m^2$ squares, where each square is assigned a label of $-1$ or $1$. It calculates the total number of possible classifiers ($2^{m^2}$) and uses Theorem 7 to state a concrete bound on the estimation error.

## Why it matters
It makes the abstract Theorem 7 concrete, showing how to count 'behaviors' and what the resulting rate looks like for a real (though simple) model.

## Active recall
> [!question]- For a grid of $m^2$ squares, what is the size of the hypothesis class $|\mathcal{H}|$?
> Since there are $m^2$ squares and each has 2 possible labels, $|\mathcal{H}| = 2^{m^2}$.

## When to use
Use when analyzing classifiers that partition the input space into disjoint regions and assign a fixed label to each region.

## Core pattern
Count the number of possible functions in the class and plug into the $\sqrt{\log(|\mathcal{H}|)/n}$ bound.

## Links
**Parent:** [[obj_048 - 1.6.2 Finite hypothesis classes]]

**Prerequisites:**
- [[obj_048 - 1.6.2 Finite hypothesis classes]]
