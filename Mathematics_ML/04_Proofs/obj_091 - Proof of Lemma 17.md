---
id: obj_091
title: "Proof of Lemma 17"
types:
  - proof
page_start: 37
page_end: 37
parent_id: "obj_090"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_090
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Lemma 17

## Conceptual overview
The proof uses the definition of the dual function as an infimum of the Lagrangian. It demonstrates that since the Lagrangian is linear in the multipliers, its infimum must be concave, and that the feasibility conditions of the primal problem force the Lagrangian to be a lower bound on the objective.

## Why it matters
It solidifies the theoretical foundation of the weak duality property.

## Active recall
> [!question]- How does the proof establish the lower bound property?
> By showing that for any feasible $w$, the terms $\xi^\top (Aw - b)$ and $\sum \nu_k f_k(w)$ in the Lagrangian are zero and non-positive respectively, making $L(w, \xi, \nu) \le f_0(w)$.

## Proves
[[obj_090 - Lemma 17]]

## Proof skeleton
**Goal.** Show the dual function is concave and provides a lower bound for the primal problem.

**Strategy.** Use the structure of the dual as a pointwise infimum and properties of the Lagrangian for feasible points.

- Observe that $(\xi, \nu) \to L(w, \xi, \nu)$ is an affine function.

- Recall that the pointwise infimum of affine functions is concave.

- For any feasible $w \in \mathcal{F}$, $Aw - b = 0$ and $f_j(w) \le 0$.

- It follows that for $\nu \ge 0$, $\xi^\top(Aw - b) + \sum \nu_i f_i(w) \le 0$, implying $L(w, \xi, \nu) \le f_0(w)$.

- Taking the infimum over $w$ on the left side preserves the inequality.

**Conclusion.** Thus $D(\xi, \nu) \le \inf_{w \in \mathcal{F}} f_0(w)$.

## Full proof with commentary
- We first prove the concavity of the dual function $D$.
   - *Why:* The Lagrange function $L(w, \xi, \nu)$ is an affine function (and thus both convex and concave) with respect to the multipliers $\xi$ and $\nu$. Since the dual function is defined as the pointwise infimum of these affine functions over the set of primal variables, it must be concave.
 - Next, we prove the weak duality lower bound property: $D(\xi, \nu) \le f_0(w)$ for any feasible $w \in \mathcal{F}$.
 - Analyze the terms of the Lagrangian for a feasible point $w$ and non-negative multipliers $\nu \ge 0$.
   - *Why:* For $w \in \mathcal{F}$, the equality constraints satisfy $Aw - b = 0$, making that term zero. The inequality constraints satisfy $f_j(w) \le 0$. Since $\nu_j \ge 0$, the product $\nu_j f_j(w)$ is non-positive.
 - Reassemble these components into the Lagrangian expression.
   - *Why:* We have $L(w, \xi, \nu) = f_0(w) + 0 + \text{non-positive terms} \le f_0(w)$.
 - Pass to the infimum over all $w$.
   - *Why:* By the definition of the dual function, $D(\xi, \nu) = \inf_{u \in \mathbb{R}^p} L(u, \xi, \nu)$. Since $D(\xi, \nu)$ is the infimum over all $u$, and we just showed that evaluation at any specific feasible $w$ is bounded by $f_0(w)$, it follows that $D(\xi, \nu) \le L(w, \xi, \nu) \le f_0(w)$.
 **Conclusion.** Thus, the dual function is concave and always underestimates the primal objective on the feasible set.

## Links
**Parent:** [[obj_090 - Lemma 17]]

**Prerequisites:**
- [[obj_090 - Lemma 17]]
