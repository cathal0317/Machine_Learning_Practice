---
id: obj_078
title: "Proof of Proposition 15"
types:
  - proof
page_start: 35
page_end: 35
parent_id: "obj_080"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_080
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 15

## Conceptual overview
The proof leverages the fact that a convex function always sits above its tangent plane. If the tangent plane is horizontal (gradient zero), the function must be at its global minimum.

## Why it matters
It proves that for convex functions, 'gradient equals zero' is not just a necessary condition, but a sufficient one.

## Active recall
> [!question]- How does the first-order convexity characterization prove sufficiency of $\nabla f = 0$?
> By substituting $\nabla f(w^*) = 0$ into $f(w) \geq f(w^*) + \langle \nabla f(w^*), w - w^* \rangle$, we immediately obtain $f(w) \geq f(w^*)$.

## Proves
[[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]

## Proof skeleton
**Goal.** Prove $w^*$ is a minimizer iff $\nabla f(w^*) = 0$.

**Strategy.** Use the necessity of zero gradient for any differentiable optimum and the first-order characterization for sufficiency.

- Necessity: By Taylor expansion, $f(w^* + tu) = f(w^*) + t \langle \nabla f(w^*), u \rangle + o(t)$. For $w^*$ to be a minimizer, the linear term must be zero in all directions.

- Sufficiency: Assume $\nabla f(w^*) = 0$. From Theorem 14, $f(w) \geq f(w^*) + \langle \nabla f(w^*), w - w^* \rangle$.

- Substituting the assumption gives $f(w) \geq f(w^*)$ for all $w$.

**Conclusion.** Thus, $w^*$ is a global minimizer.

## Full proof with commentary
- The proof handles the necessity and sufficiency of the zero-gradient condition separately. Necessity holds for any differentiable function: if $w^*$ is a minimizer, then $\nabla f(w^*) = 0$.
 - To prove sufficiency, assume that $\nabla f(w^*) = 0$ and that $f$ is convex.
 - Apply the first-order characterization of convexity (Theorem 14).
   - *Why:* For any other point $w$, convexity implies $f(w) \ge f(w^*) + \langle \nabla f(w^*), w - w^* \rangle$.
 - Substitute the assumption $\nabla f(w^*) = 0$ into this inequality.
   - *Why:* The inner product term vanishes, leaving $f(w) \ge f(w^*)$ for all $w$.
 **Conclusion.** This confirms that $w^*$ is a global minimizer, proving that the zero-gradient condition is both necessary and sufficient for optimality in the convex case.

## Links
**Parent:** [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]

**Prerequisites:**
- [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]
