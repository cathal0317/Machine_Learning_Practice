---
id: obj_015
title: "Proof of Proposition 1"
types:
  - proof
page_start: 8
page_end: 8
parent_id: "obj_012"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_012
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 1

## Conceptual overview
The proof utilizes the law of total expectation to decompose the risk $R(h)$ into a conditional expectation given $X$. It then demonstrates that for each individual value of $x$, the probability of being wrong is minimized by choosing the label that has the majority probability according to $\eta(x)$. Since this minimizes the error point-wise, it must minimize the global integrated risk.

## Why it matters
It proves that no classifier can ever beat the Bayes classifier, setting an absolute lower bound on performance.

## Active recall
> [!question]- What is the first step in the proof of Proposition 1?
> The first step is to use the law of total expectation to write $R(h) = \mathbb{E}(\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X))$.

## Proves
[[obj_012 - Proposition 1]]

## Proof skeleton
**Goal.** Show that $R(h) \geq R(h^*)$ for any hypothesis $h$.

**Strategy.** Decompose risk using conditional expectation and minimize point-wise.

- Use the tower property: $R(h) = \mathbb{E}(\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X))$.

- Expand the inner expectation: $\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X=x) = \mathbf{1}(h(x)=1)(1-\eta(x)) + \mathbf{1}(h(x)=0)\eta(x)$.

- Show that choosing $h^*(x)$ based on whether $\eta(x) > 0.5$ or $\eta(x) \leq 0.5$ yields the minimum value for this expression at every $x$.

**Conclusion.** Since the inner term is minimized for every $x$, the global expectation is minimized.

## Full proof with commentary
- The proof begins by applying the law of total expectation (the tower property) to the global misclassification risk.
   - *Why:* Since risk is defined as $R(h) = \mathbb{E}(\mathbf{1}(h(X) \neq Y))$, conditioning on the feature vector $X$ allows us to write $R(h) = \mathbb{E}(\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X))$.
 - We now evaluate the inner conditional expectation for a fixed point $X=x$.
   - *Why:* The probability of misclassification at $x$ is given by the sum of the probabilities of two mutually exclusive error events: predicting 1 when the truth is 0, and predicting 0 when the truth is 1. Using indicator functions, this is expanded as 
$$
\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X=x) = \mathbf{1}(h(x)=1)\mathbb{P}(Y=0 \mid X=x) + \mathbf{1}(h(x)=0)\mathbb{P}(Y=1 \mid X=x).
$$
 Substituting the regression function $\eta(x) = \mathbb{P}(Y=1 \mid X=x)$, we obtain $\mathbf{1}(h(x)=1)(1-\eta(x)) + \mathbf{1}(h(x)=0)\eta(x)$.
 - To minimize the global risk $R(h)$, we must choose $h(x)$ to minimize this conditional expression at every point $x$.
   - *Why:* If $\eta(x) > 1/2$, then $1 - \eta(x) < \eta(x)$, so the expression is minimized by setting $h(x)=1$. Conversely, if $\eta(x) < 1/2$, it is minimized by $h(x)=0$. At $\eta(x)=1/2$, both choices yield the same local risk.
 **Conclusion.** By definition, the Bayes classifier $h^*$ follows exactly this optimal thresholding rule, so $\mathbb{E}(\mathbf{1}(h(X) \neq Y) \mid X=x) \ge \mathbb{E}(\mathbf{1}(h^*(X) \neq Y) \mid X=x)$ for all $x$, concluding that $R(h) \ge R(h^*)$.

## Links
**Parent:** [[obj_012 - Proposition 1]]

**Prerequisites:**
- [[obj_012 - Proposition 1]]
