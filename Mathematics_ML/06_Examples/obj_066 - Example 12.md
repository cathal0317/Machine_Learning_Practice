---
id: obj_066
title: "Example 12"
types:
  - example
page_start: 29
page_end: 29
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_062
  - obj_063
  - obj_064
  - obj_065
  - obj_067
prerequisites:
  - obj_062
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 12

## Conceptual overview
Axis-aligned rectangles can shatter 4 points arranged in a diamond shape. However, they cannot shatter 5 points because the smallest rectangle containing 4 of the points must necessarily contain the 5th if that 5th point lies in the convex hull, or one point will always be 'enclosed' by the others' boundaries. Thus, the VC dimension is 4.

## Why it matters
It provides a clear visualization of how dimensionality and geometric constraints affect model capacity.

## Active recall
> [!question]- What is the VC dimension of axis-aligned rectangles in 2D?
> The VC dimension is 4.

## When to use
When evaluating simple geometric classifiers in higher dimensions.

## Core pattern
Finding a configuration that can be shattered (diamond shape) versus one that cannot (any 5 points).

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_062 - Definition 4]]
