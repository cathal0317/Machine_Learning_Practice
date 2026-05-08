---
id: obj_075
title: "Proof of Theorem 14"
types:
  - proof
page_start: 34
page_end: 34
parent_id: "obj_076"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_076
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 14

## Conceptual overview
The proof translates the algebraic definition of convexity into a statement about the function's tangent planes. It uses a limit argument on the difference quotient to show that the function always lies above its first-order Taylor approximation.

## Why it matters
It connects the geometric intuition of convexity to the calculus tools used in optimization algorithms.

## Active recall
> [!question]- How is the gradient limit used in this proof?
> By rearranging the convexity inequality and taking the limit as the step size $\lambda \to 0$, the difference quotient converges to the directional derivative, which is the inner product with the gradient.

## Proves
[[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]

## Proof skeleton
**Goal.** Show $f$ is convex iff $f(w) \geq f(v) + \nabla f(v)^\top (w-v)$.

**Strategy.** Use the definition of convexity and take the limit of a difference quotient.

- Assume $f$ is convex. Then $f((1-\lambda)v + \lambda w) \leq (1-\lambda)f(v) + \lambda f(w)$.

- Rearrange to $\frac{f(v + \lambda(w-v)) - f(v)}{\lambda} \leq f(w) - f(v)$.

- Take the limit $\lambda \to 0$. The LHS converges to $\langle \nabla f(v), w-v \rangle$.

- This yields the 'only if' part. The 'if' part is proven by weighting inequalities for two points and summing.

**Conclusion.** Therefore, the first-order condition is equivalent to convexity.

## Full proof with commentary
- We first prove the 'only if' direction: convexity implies the first-order condition. Assume $f$ is convex.
   - *Why:* For any $v, w$ and $\lambda \in (0, 1)$, we have $f((1-\lambda)v + \lambda w) \le (1-\lambda)f(v) + \lambda f(w)$.
 - Rearrange this inequality to isolate the difference quotient in terms of $\lambda$.
   - *Why:* Subtracting $f(v)$ and dividing by $\lambda$ gives 
$$
\frac{f(v + \lambda(w-v)) - f(v)}{\lambda} \le f(w) - f(v).
$$

 - Take the limit as $\lambda \to 0$.
   - *Why:* By the definition of the gradient (and the chain rule for the directional derivative), the left-hand side converges to $\langle \nabla f(v), w-v \rangle$. Thus, $\langle \nabla f(v), w-v \rangle \le f(w) - f(v)$, which is the desired condition.
 - To prove the 'if' direction, assume the first-order condition holds for all points and let $v_\lambda = (1-\lambda)v + \lambda v'$.
   - *Why:* We write the condition twice: once comparing $f(v)$ to the tangent at $v_\lambda$, and once comparing $f(v')$ to the same tangent: 
$$
f(v) \ge f(v_\lambda) + \langle \nabla f(v_\lambda), v - v_\lambda \rangle
$$
 
$$
f(v') \ge f(v_\lambda) + \langle \nabla f(v_\lambda), v' - v_\lambda \rangle.
$$

 - Multiply the first by $1-\lambda$, the second by $\lambda$, and sum them.
   - *Why:* In the sum, the inner product terms are $\langle \nabla f(v_\lambda), (1-\lambda)(v - v_\lambda) + \lambda(v' - v_\lambda) \rangle$. Substituting $v_\lambda = (1-\lambda)v + \lambda v'$, the vector in the inner product becomes exactly zero.
 **Conclusion.** The resulting sum is $(1-\lambda)f(v) + \lambda f(v') \ge f(v_\lambda)$, which is the definition of convexity.

## Links
**Parent:** [[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]

**Prerequisites:**
- [[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]
