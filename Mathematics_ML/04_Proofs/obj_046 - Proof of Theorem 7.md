---
id: obj_046
title: "Proof of Theorem 7"
types:
  - proof
page_start: 23
page_end: 23
parent_id: "obj_047"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_047
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 7

## Conceptual overview
This proof bounds the estimation error by considering the maximum deviation between empirical and true risk across the entire hypothesis class. It splits the error into a term for the optimal hypothesis in the class and a term for all others, then applies the union bound to control the maximum deviation.

## Why it matters
It explains the logarithmic dependence on the size of the hypothesis class, showing that we can handle quite large classes as long as they are finite.

## Active recall
> [!question]- Why is the union bound necessary in the proof of Theorem 7?
> Because the ERM $\hat{h}$ depends on the data, we must bound the risk for all possible hypotheses in the finite set $\mathcal{H}$ simultaneously.

## Proves
[[obj_047 - Theorem 7]]

## Proof skeleton
**Goal.** Prove $R(\hat{h}) - R(\bar{h}) \le L \sqrt{\frac{2 \log(|\mathcal{H}|) + 2 \log(\delta^{-1})}{n}}$ with high probability.

**Strategy.** Use the estimation error decomposition and apply the union bound over $\mathcal{H}$.

- Note that $R(\hat{h}) - R(\bar{h}) \le 2 \sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)|$.

- Bound the probability that any $h \in \mathcal{H}$ has a large deviation using Hoeffding's inequality.

- Apply the union bound: $\mathbb{P}(\max |R(h) - \hat{R}(h)| > t/2) \le |\mathcal{H}| \exp(-nt^2/(2L^2))$.

- Set this sum of probabilities equal to $\delta$ and solve for $t$.

**Conclusion.** The resulting bound on $t$ proves the theorem statement.

## Full proof with commentary
- We start by decomposing the estimation error using the triangle inequality relative to empirical risk.
   - *Why:* As shown in Equation 1.7, the gap $R(\hat{h}) - R(\bar{h})$ is bounded by $2 \sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)|$. Thus, to control the error with probability $1 - \delta$, we seek a value $t$ such that $\mathbb{P}(\sup_{h \in \mathcal{H}} |R(h) - \hat{R}(h)| > t/2) \le \delta$.
 - Apply the union bound over the finite hypothesis class $\mathcal{H}$.
   - *Why:* The probability that the maximum deviation exceeds $t/2$ is less than or equal to the sum of the individual deviation probabilities: $\sum_{h \in \mathcal{H}} \mathbb{P}(|R(h) - \hat{R}(h)| > t/2)$.
 - Bound each individual term in the sum using Hoeffding's inequality.
   - *Why:* Since the loss is bounded in $(0, L)$, Corollary 6 implies $\mathbb{P}(|R(h) - \hat{R}(h)| > t/2) \le 2 \exp(-2n(t/2)^2 / L^2) = 2 \exp(-nt^2 / (2L^2))$. Summing over the class gives a total bound of $2|\mathcal{H}| \exp(-nt^2 / (2L^2))$. (Note: the proof in the notes specifically handles the cases $\hat{h} = \bar{h}$ and $\hat{h} \neq \bar{h}$ to achieve a slightly tighter constant $|\mathcal{H}|$ instead of $2|\mathcal{H}|$).
 - Set the sum of probabilities equal to $\delta$ and solve for $t$.
   - *Why:* Solving $|\mathcal{H}| \exp(-nt^2 / (2L^2)) = \delta$ for $t$ yields $t^2 = 2L^2 (\log(|\mathcal{H}|) + \log(\delta^{-1})) / n$.
 **Conclusion.** Taking the square root gives the high-probability bound stated in Theorem 7.

## Links
**Parent:** [[obj_047 - Theorem 7]]

**Prerequisites:**
- [[obj_047 - Theorem 7]]
