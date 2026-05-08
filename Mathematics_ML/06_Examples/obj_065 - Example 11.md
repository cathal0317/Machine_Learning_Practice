---
id: obj_065
title: "Example 11"
types:
  - example
page_start: 28
page_end: 29
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_062
  - obj_063
  - obj_064
  - obj_066
  - obj_067
prerequisites:
  - obj_062
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 11

## Conceptual overview
Consider functions $h_{a,b}(x)$ that are 1 if $x \in (a, b)$ and 0 otherwise. For any $n$ points, the number of distinct labelings is at most $(n+1)^2$ because labels only change when the interval boundaries cross a point. For 2 points, they can be shattered, but 3 points cannot (specifically, the label sequence 1, 0, 1 is impossible), so the VC dimension is 2.

## Why it matters
It demonstrates a case where the growth function is polynomial ($n^2$) rather than exponential, leading to a finite VC dimension.

## Active recall
> [!question]- Why can't the interval class shatter 3 points on a line?
> If three points are ordered $x_1 < x_2 < x_3$, an interval cannot contain $x_1$ and $x_3$ without also containing $x_2$, making the labeling (1, 0, 1) impossible.

## When to use
When analyzing the complexity of classifiers based on single-range thresholds.

## Core pattern
Identifying impossible labelings (configurations) to establish an upper bound on VC dimension.

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_062 - Definition 4]]
