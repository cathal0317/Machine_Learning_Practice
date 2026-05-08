---
id: obj_027
title: "Example 4"
types:
  - example
page_start: 16
page_end: 18
parent_id: "obj_028"
children_ids: []
sibling_ids:
  - obj_025
  - obj_026
  - obj_029
prerequisites:
  - obj_018
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example 4

## Conceptual overview
This example analyzes the learning of axis-aligned rectangles in $\mathbb{R}^2$ using a consistent learner (the smallest enclosing rectangle). It demonstrates a PAC-style guarantee, bounding the sample size needed to ensure small risk with high probability.

## Why it matters
It provides a concrete sample complexity result—$n \ge \frac{4}{\epsilon} \log(4/\delta)$—for a simple geometric hypothesis class.

## Active recall
> [!question]- In the learning rectangles example, what is the required sample size $n$ to achieve risk $\epsilon$ with confidence $1-\delta$?
> $n \ge \frac{4}{\epsilon} \log(4/\delta)$.

## When to use
When trying to derive simple PAC (Probably Approximately Correct) bounds for geometric indicator functions.

## Core pattern
Use the union bound to control the probability that any significant region of the target set is omitted from the sample.

## Links
**Parent:** [[obj_028 - 1.5 Excess risk]]

**Prerequisites:**
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
