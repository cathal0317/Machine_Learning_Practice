---
id: obj_004
title: "The loss function"
types:
  - definition
page_start: 5
page_end: 5
parent_id: "obj_005"
children_ids: []
sibling_ids:
  - obj_003
  - obj_007
  - obj_008
  - obj_009
prerequisites: []
used_in:
  - obj_009
  - obj_018
  - obj_141
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# The loss function

## Conceptual overview
A loss function $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$ provides a quantitative measure of the 'cost' or 'error' incurred when a hypothesis $h$ predicts an output value for a given true value $y$. Different tasks use different standard losses, such as the squared error for regression and the unit (0-1) loss for classification. The choice of loss function is critical because the learning process aims to minimize the expectation of this loss.

## Why it matters
The loss function is the mathematical objective that defines what 'good' prediction looks like for a specific model.

## Active recall
> [!question]- What is the standard loss function used for regression in these notes?
> The standard regression loss is the squared error: $\ell(h(x), y) = (h(x) - y)^2$.

> [!question]- Define the unit loss function for classification.
> The unit loss is defined as $\ell(h(x), y) = 0$ if $h(x) = y$ and $1$ if $h(x) \neq y$.

## Mental picture
Think of the loss function as a penalty score: it is zero when you are exactly right and grows larger as your prediction becomes more 'wrong'.

## Common confusions
Confusing the loss (error on a single point) with the risk (average error over the whole distribution).

## Links
**Parent:** [[obj_005 - 1.1 Classification and regression]]

**Used in:**
- [[obj_009 - The Risk]]
- [[obj_018 - 1.3 Empirical risk minimization and hypothesis classes]]
- [[obj_141 - 3.2 Cross-entropy loss]]
