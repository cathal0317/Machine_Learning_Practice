---
id: obj_059
title: "Proof of Lemma 10"
types:
  - proof
page_start: 27
page_end: 27
parent_id: "obj_058"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_058
  - obj_036
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Lemma 10

## Conceptual overview
The proof defines Rademacher averages as a set of random variables. It then identifies these averages as sub-Gaussian and uses the property that the expectation of the maximum of $d$ sub-Gaussian variables grows as $\sqrt{\log d}$.

## Why it matters
It explains why finite behaviors result in logarithmic complexity growth, which is the mathematical reason why large (but not too complex) models work.

## Active recall
> [!question]- What sub-Gaussian parameter is used for the Rademacher averages in this proof?
> The parameter is $n^{-1/2}$.

## Proves
[[obj_058 - Lemma 10. [Massart’s Lemma]]]

## Proof skeleton
**Goal.** Prove $\hat{\mathcal{R}} \le \sqrt{2 \log d / n}$ where $d = |\mathcal{F}(z_{1:n})|$.

**Strategy.** Identify Rademacher averages as sub-Gaussian and bound their maximum.

- Define $W_j = \frac{1}{n} \sum \varepsilon_i f_j(z_i)$ for each behavior $f_j$.

- Show $W_j$ is sub-Gaussian with parameter $(\sum 1/n^2)^{1/2} = n^{-1/2}$.

- Use Proposition 11 to bound the expectation of the maximum $\mathbb{E}(\max W_j) \le n^{-1/2} \sqrt{2 \log d}$.

**Conclusion.** The resulting bound matches the statement of Massart's Lemma.

## Full proof with commentary
- Let $d = |\mathcal{F}(z_{1:n})|$ denote the number of distinct behaviors of the function class on the dataset. We label these behaviors as $f_1, \dots, f_d$.
 - Define a set of random variables $W_j = \frac{1}{n} \sum_{i=1}^n \varepsilon_i f_j(z_i)$, which represent the correlation of each behavior with random noise.
   - *Why:* By the definition of empirical Rademacher complexity, $\hat{\mathcal{R}}(\mathcal{F}(z_{1:n})) = \mathbb{E}(\max_j W_j)$.
 - We show that each $W_j$ is sub-Gaussian.
   - *Why:* Each term $\varepsilon_i f_j(z_i)$ is sub-Gaussian with parameter $f_j(z_i)$. By Proposition 5, their average $W_j$ is sub-Gaussian with parameter $(\sum f_j(z_i)^2 / n^2)^{1/2}$. Since $|f| \le 1$, this parameter is bounded by $(\sum 1/n^2)^{1/2} = n^{-1/2}$.
 - Apply Proposition 11 to bound the expectation of the maximum of these $d$ sub-Gaussian variables.
   - *Why:* Proposition 11 states that for $d$ variables with sub-Gaussian parameter $\sigma$, $\mathbb{E}(\max W_j) \le \sigma \sqrt{2 \log d}$.
 **Conclusion.** Substituting $\sigma = n^{-1/2}$ and $d = |\mathcal{F}(z_{1:n})|$ yields the stated bound $\sqrt{2 \log(|\mathcal{F}(z_{1:n})|) / n}$.

## Links
**Parent:** [[obj_058 - Lemma 10. [Massart’s Lemma]]]

**Prerequisites:**
- [[obj_058 - Lemma 10. [Massart’s Lemma]]]
- [[obj_036 - Definition 2]]
