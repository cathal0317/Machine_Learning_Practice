---
id: obj_139
title: "Remark 18"
types:
  - remark
page_start: 57
page_end: 57
parent_id: "obj_138"
children_ids: []
sibling_ids:
  - obj_140
prerequisites:
  - obj_140
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Remark 18

## Conceptual overview
Demonstrates that the Universal Approximation Theorem applies to ReLU networks by showing that ReLU functions can be combined to approximate sigmoidal 'non-decreasing continuous' functions required by the original theorem.

## Why it matters
It bridges the gap between abstract theorems for sigmoidal functions and the modern practical preference for ReLU activations.

## Active recall
> [!question]- How can a ReLU network approximate a sigmoidal function for the purposes of the theorem?
> By defining $\sigma_1(r) = \text{ReLU}(z) - \text{ReLU}(z-1)$, which acts as a ramp function that is 0 for $r < 0$ and 1 for $r > 1$.

## Mental picture
Imagine ReLU ramps being combined to build a staircase that mimics any continuous curve, similar to how Riemann sums approximate an integral.

## Common confusions
Assuming the Universal Approximation Theorem only applies to functions that explicitly vanish at negative infinity and approach one at positive infinity.

## Links
**Parent:** [[obj_138 - 3.1.2 Universal approximation]]

**Prerequisites:**
- [[obj_140 - Theorem 24 (The Universal Approximation theorem)]]
