---
id: obj_178
title: "Interpretation of ELBO terms"
types:
  - discussion
page_start: 76
page_end: 76
parent_id: "obj_174"
children_ids: []
sibling_ids:
  - obj_173
  - obj_175
  - obj_176
  - obj_180
  - obj_181
prerequisites:
  - obj_175
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Interpretation of ELBO terms

## Conceptual overview
This discussion interprets the two parts of the ELBO: the first term acts as a reconstruction accuracy measure, while the second (the KL term) acts as a regularizer forcing the latent space to match a prior (usually Gaussian).

## Why it matters
Understanding this trade-off is critical for tuning VAEs to ensure they neither ignore the latent code (posterior collapse) nor simply memorize the data.

## Active recall
> [!question]- What effect does the term $-\frac{1}{2} ||F_\psi(x)||^2$ have on the latent space?
> It penalizes large latent codes, encouraging the representation to stay close to the origin in the latent space.

## Narrative flow
Analyzes the two terms of the ELBO objective under Gaussian assumptions, explaining how each term shapes the latent space representation.

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Prerequisites:**
- [[obj_175 - Proposition 26 (ELBO)]]
