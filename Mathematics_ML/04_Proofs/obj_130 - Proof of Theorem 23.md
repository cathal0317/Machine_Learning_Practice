---
id: obj_130
title: "Proof of Theorem 23"
types:
  - proof
page_start: 53
page_end: 54
parent_id: "obj_129"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_129
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 23

## Conceptual overview
The proof establish a recursive inequality for the expected squared error at each step. By using the unbiasedness of the stochastic gradient and the strong convexity of the objective, it shows that each update contracts the expected distance to the optimum, plus a term representing sampling noise.

## Why it matters
This proof demonstrates how to handle randomness in optimization analysis using the tower property of expectations and iterative inequalities.

## Active recall
> [!question]- How is the randomness of the sampled index handled in the proof of Theorem 23?
> The proof uses conditional expectation $\mathbb{E}_k$ (conditioned on all previous iterates) to leverage the property that $\mathbb{E}_k(\nabla f_{i_k}(w_k)) = \nabla f(w_k)$.

> [!question]- What role does the assumption $\|\nabla f_i(w)\|^2 \le C^2$ play in this proof?
> It bounds the second moment of the stochastic gradient, ensuring the 'noise' term in the recursion does not grow uncontrollably.

## Proves
[[obj_129 - Theorem 23]]

## Proof skeleton
**Goal.** Prove that $\mathbb{E}\|w_k - w^*\|^2$ decays at an $O(1/k)$ rate.

**Strategy.** Expand the squared norm of the error update, take expectations, and use strong convexity to derive a recursion.

- Expand $\mathbb{E}_k \|w_{k+1} - w^*\|^2 = \mathbb{E}_k \|w_k - \tau_k \nabla f_{i_k}(w_k) - w^*\|^2$.

- Distribute the norm into $\|w_k - w^*\|^2 - 2\tau_k \langle w_k - w^*, \mathbb{E}_k \nabla f_{i_k}(w_k) \rangle + \tau_k^2 \mathbb{E}_k \|\nabla f_{i_k}(w_k)\|^2$.

- Substitute $\nabla f(w_k)$ for the expectation and use the bounded gradient assumption $C^2$.

- Use the strong convexity inequality $\langle \nabla f(w_k), w_k - w^* \rangle \ge \mu \|w_k - w^*\|^2$ to bound the inner product term.

- Derive the recursion $\mathbb{E}\|w_{k+1} - w^*\|^2 \le (1 - 2\tau_k \mu) \mathbb{E}\|w_k - w^*\|^2 + \tau_k^2 C^2$.

- Use induction with the specific stepsize $\tau_k = 1/(\mu(k+1))$.

**Conclusion.** The recursion yields $\mathbb{E}\|w_{k+1} - w^*\|^2 \le R/(k+2)$.

## Full proof with commentary
- We analyze the expected squared distance to the optimum by expanding the recursive update.
   - *Why:* Letting $\mathbb{E}_k$ denote expectation conditioned on the history up to step $k$, we have 
$$
\mathbb{E}_k \|w_{k+1} - w^*\|^2 = \mathbb{E}_k \|w_k - \tau_k \nabla f_{i_k}(w_k) - w^*\|^2.
$$

 - Expand the squared norm into three terms.
   - *Why:* 
$$
\|w_k - w^*\|^2 - 2\tau_k \langle w_k - w^*, \mathbb{E}_k \nabla f_{i_k}(w_k) \rangle + \tau_k^2 \mathbb{E}_k \|\nabla f_{i_k}(w_k)\|^2.
$$

 - Utilize the unbiasedness of the stochastic gradient and the bounded gradient assumption.
   - *Why:* Since $\mathbb{E}_k(\nabla f_{i_k}(w_k)) = \nabla f(w_k)$ and $\|\nabla f_i\|^2 \le C^2$, the expression is bounded by $\|w_k - w^*\|^2 - 2\tau_k \langle w_k - w^*, \nabla f(w_k) \rangle + \tau_k^2 C^2$.
 - Apply the definition of $\mu$-strong convexity.
   - *Why:* For the convex objective $f$, $\langle \nabla f(w_k), w_k - w^* \rangle \ge \mu \|w_k - w^*\|^2$. Substituting this gives the recursion: 
$$
\mathbb{E}_k \|w_{k+1} - w^*\|^2 \le (1 - 2\tau_k \mu) \|w_k - w^*\|^2 + \tau_k^2 C^2.
$$

 - Take the total expectation and prove the $O(1/k)$ decay via induction.
   - *Why:* We define $\epsilon_k = \mathbb{E} \|w_k - w^*\|^2$. Assume $\epsilon_k \le R/(k+1)$. With $\tau_k = 1/(\mu(k+1))$, the recurrence becomes 
$$
\epsilon_{k+1} \le (1 - \frac{2}{k+1}) \frac{R}{k+1} + \frac{1}{\mu^2(k+1)^2} C^2.
$$

 **Conclusion.** Algebraic simplification shows that the right-hand side is bounded by $R/(k+2)$, completing the induction and proving the $O(1/k)$ rate.

## Links
**Parent:** [[obj_129 - Theorem 23]]

**Prerequisites:**
- [[obj_129 - Theorem 23]]
