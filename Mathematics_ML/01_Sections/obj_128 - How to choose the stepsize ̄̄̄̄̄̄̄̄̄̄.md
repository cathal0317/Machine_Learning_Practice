---
id: obj_128
title: "How to choose the stepsize ̄̄̄̄̄̄̄̄̄̄"
types:
  - discussion
page_start: 53
page_end: 53
parent_id: "obj_127"
children_ids: []
sibling_ids:
  - obj_124
  - obj_125
  - obj_126
  - obj_129
  - obj_131
  - obj_132
prerequisites:
  - obj_126
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# How to choose the stepsize ̄̄̄̄̄̄̄̄̄̄

## Conceptual overview
This discussion addresses the need for a decaying stepsize schedule in SGD. Unlike GD, where a fixed stepsize can work for smooth functions, SGD requires the stepsize to eventually vanish to counteract the inherent noise from stochastic sampling and ensure convergence to the exact minimum.

## Why it matters
A fixed stepsize in SGD leads to the algorithm 'wandering' around the optimum without ever reaching it; decaying stepsizes are required for asymptotic convergence.

## Active recall
> [!question]- Why must the stepsize $\tau_k$ in SGD eventually converge to 0?
> It must converge to 0 to cancel out the 'noise' induced by stochastic sampling from individual data points.

> [!question]- What is a common stepsize schedule used for SGD as described in the notes?
> A common schedule is $\tau_k = \tau_0 / (1 + k/k_0)$, where $k_0$ serves as a warm-up phase parameter.

## Narrative flow
The discussion focuses on the trade-off between converging fast and cancelling noise, concluding with the recommendation for $1/k$ asymptotic schedules.

## Links
**Parent:** [[obj_127 - 2.5 Stochastic gradient descent]]

**Prerequisites:**
- [[obj_126 - Stochastic gradient descent (SGD) Algorithm]]
