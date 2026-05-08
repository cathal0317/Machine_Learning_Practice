---
id: obj_175
title: "Proposition 26 (ELBO)"
types:
  - proposition
page_start: 75
page_end: 75
parent_id: "obj_174"
children_ids:
  - obj_177
sibling_ids:
  - obj_173
  - obj_176
  - obj_178
  - obj_180
  - obj_181
prerequisites: []
used_in:
  - obj_177
  - obj_180
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Proposition 26 (ELBO)

## Conceptual overview
The Evidence Lower Bound (ELBO) provides a lower bound on the log-likelihood of data under a latent variable model. It allows for optimization w.r.t. parameters by maximizing this tractable bound instead of the intractable likelihood.

## Why it matters
It is the fundamental objective function for all variational inference methods in deep learning.

## Active recall
> [!question]- Under what condition does equality hold in the ELBO inequality?
> Equality holds when the approximate posterior $q$ equals the true posterior distribution $p(z|x)$.

## Exact statement
Let $q$ be any distribution on $\mathcal{Z}$. Then 
$$
\log(p_\theta(x)) \geqslant \mathbb{E}_{z \sim q}(\log(p_\theta(x|z))) - KL(q || p_Z)
$$
 where $KL(q || p_Z) = \mathbb{E}_q(\log(q/p_Z))$.

## Links
**Parent:** [[obj_174 - 3.6.2 Variational autoencoders]]

**Children:**
- [[obj_177 - Proof of Proposition 26]]

**Used in:**
- [[obj_177 - Proof of Proposition 26]]
- [[obj_180 - Maximizing the ELBO]]
