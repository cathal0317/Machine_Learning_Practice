---
id: obj_109
title: "Choosing the stepsize"
types:
  - discussion
page_start: 46
page_end: 47
parent_id: "obj_110"
children_ids: []
sibling_ids:
  - obj_108
  - obj_111
  - obj_112
  - obj_118
  - obj_122
prerequisites:
  - obj_108
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Choosing the stepsize

## Conceptual overview
The performance of gradient descent depends heavily on the stepsize $\tau_k$. Strategies include 'greedy choice' (minimizing the function along the search direction) which can lead to zig-zagging, and the 'Armijo rule' (backtracking line search) which provides a 'sufficiently large' decay guarantee.

## Why it matters
Poor stepsize selection can cause gradient descent to oscillate or diverge, making robust selection rules vital for performance.

## Active recall
> [!question]- What is the Armijo rule?
> The Armijo rule (or backtracking line search) is a strategy to choose a stepsize that guarantees a sufficient decrease in the function value by checking the inequality $f(w_k + \tau d_k) \le f(w_k) + \alpha \tau \langle d_k, \nabla f(w_k) \rangle$.

## Narrative flow
The discussion introduces the problem of stepsize selection, then explores the 'greedy' line search and its geometric drawbacks, and finally presents the Armijo rule as a robust alternative.

## Links
**Parent:** [[obj_110 - 2.4 Gradient descent]]

**Prerequisites:**
- [[obj_108 - Gradient descent algorithm]]
