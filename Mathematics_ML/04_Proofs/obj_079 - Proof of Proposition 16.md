---
id: obj_079
title: "Proof of Proposition 16"
types:
  - proof
page_start: 35
page_end: 35
parent_id: "obj_081"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_081
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 16

## Conceptual overview
The proof shows that at a constrained minimum, any direction pointing into the feasible set must have a non-negative directional derivative. If it were negative, one could move a tiny bit into the set and improve the function value, contradicting optimality.

## Why it matters
It explains why we don't necessarily have a zero gradient at the boundary of a constrained set.

## Active recall
> [!question]- Why is the directional derivative non-negative at a constrained minimum?
> If it were negative, moving a small distance $\lambda$ in that direction (which stays in the convex feasible set) would yield a lower function value.

## Proves
[[obj_081 - Proposition 16 (First order optimality condition - constrained)]]

## Proof skeleton
**Goal.** Prove $w^*$ is a constrained minimizer iff $\nabla f(w^*)^\top (w - w^*) \geq 0$ for all $w \in \mathcal{F}$.

**Strategy.** Use the convexity of the set and function with a directional derivative argument.

- If the condition holds, $f(w) \geq f(w^*) + \langle \nabla f(w^*), w - w^* \rangle \geq f(w^*)$ by convexity.

- For the converse, assume $w^*$ is a minimizer but there exists $w$ such that the inner product is negative.

- Define the feasible path $v(\lambda) = (1-\lambda)w^* + \lambda w$. The derivative of $f(v(\lambda))$ at 0 is negative.

- For small $\lambda$, $f(v(\lambda)) < f(v(0))$, which contradicts optimality.

**Conclusion.** Thus, the first-order condition is necessary and sufficient.

## Full proof with commentary
- We first prove sufficiency: assume $\langle \nabla f(w^*), w - w^* \rangle \ge 0$ for all $w \in \mathcal{F}$.
   - *Why:* By the first-order characterization of convexity, we have $f(w) \ge f(w^*) + \langle \nabla f(w^*), w - w^* \rangle$. Given our assumption, the second term is non-negative, so $f(w) \ge f(w^*)$, and $w^*$ is a global minimizer.
 - Now we prove necessity by contradiction. Assume $w^*$ is a minimizer but there exists some $w \in \mathcal{F}$ such that $\langle \nabla f(w^*), w - w^* \rangle < 0$.
 - Construct a feasible path $v(\lambda) = (1-\lambda)w^* + \lambda w$ for $\lambda \in (0, 1)$.
   - *Why:* Since $\mathcal{F}$ is a convex set, this path is entirely contained within the feasible region.
 - Define $g(\lambda) = f(v(\lambda))$ and compute its derivative at $\lambda = 0$.
   - *Why:* By the chain rule, $g'(0) = \langle \nabla f(w^*), w - w^* \rangle$. Our assumption implies $g'(0) < 0$.
 - Observe that since the derivative is negative, the function must be decreasing for small $\lambda$.
   - *Why:* For small $\lambda$, $f(v(\lambda)) < f(v(0)) = f(w^*)$, which contradicts the fact that $w^*$ is a minimizer.
 **Conclusion.** Therefore, the non-negative inner product condition is a necessary and sufficient characterization of constrained optima for convex functions.

## Links
**Parent:** [[obj_081 - Proposition 16 (First order optimality condition - constrained)]]

**Prerequisites:**
- [[obj_081 - Proposition 16 (First order optimality condition - constrained)]]
