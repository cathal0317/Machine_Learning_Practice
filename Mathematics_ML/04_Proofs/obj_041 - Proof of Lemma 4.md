---
id: obj_041
title: "Proof of Lemma 4"
types:
  - proof
page_start: 22
page_end: 22
parent_id: "obj_040"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_040
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Lemma 4

## Conceptual overview
This proof establishes that a bounded random variable is sub-Gaussian by using the technique of symmetrization. It introduces an independent copy and a Rademacher random variable to create a symmetric difference, allowing the use of Jensen's inequality to bound the moment generating function.

## Why it matters
Hoeffding's Lemma is the technical foundation for Hoeffding's Inequality, which is one of the most important tools for bounding generalization error in machine learning.

## Active recall
> [!question]- What is the role of the Rademacher random variable in the proof of Hoeffding's Lemma?
> It is used in the symmetrization step to represent the distribution of the difference $W - W'$, allowing the expectation to be bounded by the sub-Gaussian property of Rademacher variables.

> [!question]- Why is Jensen's conditional inequality applied in this proof?
> It is used to bound the moment generating function $’\mathbb{E}(e^{\alpha W})’$ by the expectation of the exponential of the difference $’\mathbb{E}(e^{\alpha(W-W')})’$.

## Proves
[[obj_040 - Lemma 4 (Hoeffding’s lemma)]]

## Proof skeleton
**Goal.** Prove that if $W \in (a, b)$, then $\mathbb{E}(e^{\alpha(W-\mathbb{E}(W))}) \le e^{\alpha^2(b-a)^2/2}$.

**Strategy.** Use symmetrization by introducing an independent copy $W'$ and a Rademacher variable $\varepsilon$.

- Assume $\mathbb{E}(W)=0$ and write $\mathbb{E}(e^{\alpha W}) = \mathbb{E}(e^{\alpha(W-\mathbb{E}(W'))})$.

- Apply Jensen's conditional inequality to show $\mathbb{E}(e^{\alpha W}) \le \mathbb{E}(e^{\alpha(W-W')})$.

- Note that $W - W'$ has the same distribution as $\varepsilon(W - W')$ where $\varepsilon$ is Rademacher.

- Use the fact that $\varepsilon$ is sub-Gaussian with parameter 1 and $|W-W'| \le b-a$ to bound the expectation.

**Conclusion.** The MGF is bounded by $e^{\alpha^2(b-a)^2/2}$.

## Full proof with commentary
- We assume without loss of generality that $\mathbb{E}(W) = 0$ and introduce an independent copy $W'$ of $W$ to facilitate symmetrization.
   - *Why:* Since $\mathbb{E}(W') = 0$, we can write the moment generating function as $\mathbb{E}(e^{\alpha W}) = \mathbb{E}(e^{\alpha(W - \mathbb{E}(W'))})$.
 - Apply the conditional version of Jensen's inequality to move the inner expectation outside the exponential.
   - *Why:* Using the convexity of the exponential function, we have $\mathbb{E}(e^{\alpha(W - \mathbb{E}(W' \mid W))}) \le \mathbb{E}(\mathbb{E}(e^{\alpha(W - W')} \mid W)) = \mathbb{E}(e^{\alpha(W - W')})$.
 - Note that the symmetric variable $W - W'$ has the same distribution as $\varepsilon(W - W')$ where $\varepsilon$ is an independent Rademacher variable.
   - *Why:* Since $W$ and $W'$ are i.i.d., $W - W'$ and $W' - W$ share the same distribution. Thus, multiplying by a random sign $\varepsilon$ does not change the expectation.
 - We now take the expectation over $\varepsilon$ first, using the fact that Rademacher variables are sub-Gaussian with parameter 1.
   - *Why:* For fixed $W$ and $W'$, the conditional expectation $\mathbb{E}_{\varepsilon}(e^{\alpha \varepsilon (W - W')})$ is bounded by $e^{\alpha^2(W - W')^2 / 2}$.
 - Finally, substitute the boundedness assumption $|W - W'| \le b - a$.
   - *Why:* Integrating out the remaining variables, we find the MGF is bounded by $e^{\alpha^2(b - a)^2 / 2}$.
 **Conclusion.** This confirms that any bounded variable is sub-Gaussian with parameter proportional to its range.

## Links
**Parent:** [[obj_040 - Lemma 4 (Hoeffding’s lemma)]]

**Prerequisites:**
- [[obj_040 - Lemma 4 (Hoeffding’s lemma)]]
