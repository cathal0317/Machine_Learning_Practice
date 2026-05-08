---
id: obj_045
title: "Proof of Corollary 6"
types:
  - proof
page_start: 23
page_end: 23
parent_id: "obj_044"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_044
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Corollary 6

## Conceptual overview
The proof treats the sum of bounded variables as a sum of independent sub-Gaussian variables. It calculates the resulting sub-Gaussian parameter and applies the general tail bound for sub-Gaussian variables.

## Why it matters
It demonstrates how high-level concentration inequalities are built from simpler components like MGF bounds and independence properties.

## Active recall
> [!question]- What previously proved results are used in the proof of Hoeffding's Inequality?
> Hoeffding's Lemma (Lemma 4) to establish sub-Gaussianity of bounded variables, Proposition 5 for the parameter of the sum, and Proposition 3 for the final tail bound.

## Proves
[[obj_044 - Corollary 6 (Hoeffding’s inequality)]]

## Proof skeleton
**Goal.** Derive the probability bound for the empirical average of independent bounded variables.

**Strategy.** Map the bounded variables to sub-Gaussian variables and use the aggregate tail bound.

- Apply Hoeffding's Lemma to show each $W_i$ is sub-Gaussian with parameter $(b_i - a_i)/2$.

- Use Proposition 5 to determine that $Z = \frac{1}{n} \sum W_i$ is sub-Gaussian with parameter $n^{-1}(\sum (b_i-a_i)^2/4)^{1/2}$.

- Apply the sub-Gaussian tail bound from Proposition 3 using this aggregated parameter.

**Conclusion.** The resulting exponential bound matches the statement of Corollary 6.

## Full proof with commentary
- The proof begins by characterizing the sub-Gaussian properties of each individual bounded variable $W_i$.
   - *Why:* By Hoeffding's Lemma (Lemma 4), since $a_i \le W_i \le b_i$, each $W_i$ is sub-Gaussian with parameter $\sigma_i = (b_i - a_i)/2$.
 - We now consider the average $Z = \frac{1}{n} \sum W_i$ as a linear combination with weights $\gamma_i = 1/n$.
   - *Why:* Invoking Proposition 5, the sum of independent sub-Gaussian variables is also sub-Gaussian. The aggregated parameter for $Z$ is 
$$
\sigma_Z = \left( \sum_{i=1}^n \left( \frac{1}{n} \right)^2 \frac{(b_i - a_i)^2}{4} \right)^{1/2} = \frac{1}{2n} \left( \sum_{i=1}^n (b_i - a_i)^2 \right)^{1/2}.
$$

 - Apply the general tail bound for sub-Gaussian variables to this aggregated variable $Z$.
   - *Why:* Substituting the calculated parameter $\sigma_Z$ into the bound from Proposition 3, $\mathbb{P}(Z - \mathbb{E}(Z) \ge t) \le \exp(-t^2 / (2\sigma_Z^2))$.
 - Simplify the denominator in the exponent.
   - *Why:* We have $2\sigma_Z^2 = 2 \frac{1}{4n^2} \sum (b_i - a_i)^2 = \frac{1}{2n^2} \sum (b_i - a_i)^2$, so $1 / (2\sigma_Z^2) = 2n^2 / \sum (b_i - a_i)^2$.
 **Conclusion.** This yields the one-sided bound in equation 1.13; the two-sided bound follows by applying the same logic to $-Z$ and using the union bound.

## Links
**Parent:** [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]

**Prerequisites:**
- [[obj_044 - Corollary 6 (Hoeffding’s inequality)]]
