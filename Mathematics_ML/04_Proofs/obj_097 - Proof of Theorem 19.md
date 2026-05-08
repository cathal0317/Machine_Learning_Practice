---
id: obj_097
title: "Proof of Theorem 19"
types:
  - proof
page_start: 39
page_end: 39
parent_id: "obj_096"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_096
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 19

## Conceptual overview
The proof establishes that a point satisfying the KKT conditions minimizes the Lagrangian, and that the value of this minimum equals the primal objective value at that point, thus satisfying the definition of optimality under strong duality.

## Why it matters
It validates the use of KKT conditions as a definitive test for optimality in convex optimization.

## Active recall
> [!question]- How does the stationary condition help in the proof of Theorem 19?
> The stationary condition $\partial_w L(w, \xi^*, \nu^*) = 0$ implies that $w^*$ is a global minimizer of the convex function $w \to L(w, \xi^*, \nu^*)$.

## Proves
[[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]

## Proof skeleton
**Goal.** Show that points satisfying KKT conditions are optimal under strong duality.

**Strategy.** Use the stationarity condition to identify the Lagrangian minimizer and verify the dual value matches the primal.

- Note that the stationary condition $\partial_w L(w, \xi^*, \nu^*) = 0$ implies $w^*$ minimizes the convex Lagrangian.

- By definition of the dual function, $D(\xi^*, \nu^*) = L(w^*, \xi^*, \nu^*)$.

- Expand the Lagrangian at $w^*$ to get $f_0(w^*) + \xi^{*\top}(Aw^* - b) + \sum \nu^*_i f_i(w^*)$.

- Apply primal feasibility ($Aw^* - b = 0$) and complementary slackness ($\sum \nu^*_i f_i(w^*) = 0$).

**Conclusion.** The resulting equality $D(\xi^*, \nu^*) = f_0(w^*)$ establishes that the pair is primal-dual optimal.

## Full proof with commentary
- The proof aims to show that any point satisfying the KKT conditions provides equal primal and dual values, establishing optimality.
 - Consider the Lagrangian $w \to L(w, \xi^*, \nu^*)$ and apply the stationary condition.
   - *Why:* The KKT stationary condition $\partial_w L(w, \xi^*, \nu^*) = 0$ implies that $w^*$ is the global minimizer of this convex function. Therefore, the dual function value is exactly $D(\xi^*, \nu^*) = L(w^*, \xi^*, \nu^*)$.
 - Expand the expression for the Lagrangian at the point $w^*$.
   - *Why:* We have 
$$
L(w^*, \xi^*, \nu^*) = f_0(w^*) + \langle \xi^*, Aw^* - b \rangle + \sum_{i=1}^m \nu_i^* f_i(w^*).
$$

 - Substitute the KKT conditions for primal feasibility and complementary slackness into this expression.
   - *Why:* Primal feasibility implies $Aw^* - b = 0$, so the second term vanishes. Complementary slackness requires each $\nu_i^* f_i(w^*) = 0$, so the third term also vanishes.
 - Compare the resulting values.
   - *Why:* We are left with $D(\xi^*, \nu^*) = f_0(w^*)$.
 **Conclusion.** Since the dual value matches the primal objective value at these points, the duality gap is zero, which characterizes a primal-dual optimal pair under strong duality.

## Links
**Parent:** [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]

**Prerequisites:**
- [[obj_096 - Theorem 19 (Karush-Kuhn-Tucker)]]
