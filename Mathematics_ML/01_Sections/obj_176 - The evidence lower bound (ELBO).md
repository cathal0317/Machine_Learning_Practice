---
id: obj_176
title: "The evidence lower bound (ELBO)"
types:
  - discussion
page_start: 75
page_end: 75
parent_id: "obj_174"
children_ids: []
sibling_ids:
  - obj_173
  - obj_175
  - obj_178
  - obj_180
  - obj_181
prerequisites: []
used_in:
  - obj_174
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# The evidence lower bound (ELBO)

## Conceptual overview
The discussion explains that maximizing the ELBO is a practical alternative to directly minimizing KL divergence between the data and the model distribution. It utilizes Jensen's inequality to move expectations outside of logarithms.

## Why it matters
It simplifies the optimization landscape of VAEs, making them solvable with stochastic gradient methods.

## Active recall
> [!question]- What is the 'annoying thing' about maximizing the log-likelihood $p_\theta$ directly?
> Since $p_\theta$ is defined as an expectation $\mathbb{E}_z$, there is an expectation inside the log, which is difficult to optimize. Moving the log inside via Jensen's inequality creates a tractable bound.

## Narrative flow
Identifies the mathematical friction in standard MLE, proposes Jensen's inequality as the tool to solve it, and introduces the term 'Evidence Lower Bound'.

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Used in:**
- [[obj_174 - 3.6.2 Variational autoencoders]]
