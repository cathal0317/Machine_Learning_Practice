---
id: obj_123
title: "Proof of Theorem 22"
types:
  - proof
page_start: 51
page_end: 52
parent_id: "obj_121"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_121
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 22

## Conceptual overview
The proof first uses the smoothness property to show that the function value decreases at every step of gradient descent. It then uses the 3-point inequality and summation over iterations to derive the $1/T$ rate for smooth functions and a contraction argument for the linear rate in the strongly convex case.

## Why it matters
The techniques used in this proof (the 3-point inequality and iterative summation) are fundamental to the analysis of almost all first-order optimization algorithms.

## Active recall
> [!question]- What is the significance of showing that $f(w_k)$ is a strictly decreasing sequence in the proof?
> It ensures that every step of the algorithm makes progress towards the minimum, which is necessary for establishing the convergence of the sequence iterates.

> [!question]- How is the $O(1/k)$ rate established for smooth functions?
> By summing the progress made in each step $f(w_{k+1}) - f(w^*)$ and applying the 3-point inequality, we arrive at a bound $f(w_T) - f(w^*) \le (L/2T) \|w_0 - w^*\|^2$.

## Proves
[[obj_121 - Theorem 22]]

## Proof skeleton
**Goal.** Prove sublinear $O(1/k)$ objective convergence for smooth functions and linear iterate convergence for strongly convex functions.

**Strategy.** Use the smoothness upper bound to show descent, then use the 3-point inequality to sum progress over iterations.

- Start with the smoothness inequality $f(w_{k+1}) - f(u) \le f(w_k) - f(u) + \langle \nabla f(w_k), w_{k+1} - w_k \rangle + \frac{L}{2}\|w_k - w_{k+1}\|^2$.

- Use convexity to bound $f(w_k) - f(u) \le \langle \nabla f(w_k), w_k - u \rangle$.

- Combine and simplify using the definition $w_{k+1} - w_k = -\tau \nabla f(w_k)$ with $\tau = 1/L$ to get $f(w_{k+1}) - f(u) \le L \langle w_k - w_{k+1}, w_{k+1} - u \rangle + \frac{L}{2}\|w_k - w_{k+1}\|^2$.

- Apply the 3-point inequality to the inner product term and sum from $k=0$ to $T-1$ with $u = w^*$.

- Bound the sum by $T(f(w_T) - f(w^*))$ to yield the $O(1/T)$ rate.

- For the linear rate, use strong convexity $\frac{\mu}{2}\|w_{k+1} - w^*\|^2 \le f(w_{k+1}) - f(w^*)$ and combine with the descent bound.

**Conclusion.** Rearrange to show $\|w_{k+1} - w^*\| \le \rho \|w_k - w^*\|$ with $\rho < 1$.

## Full proof with commentary
- We prove the $O(1/k)$ rate for smooth functions by assuming a fixed stepsize $\tau = 1/L$. First, use the quadratic upper bound from Proposition 21.
   - *Why:* For the step $w_{k+1} = w_k - \frac{1}{L} \nabla f(w_k)$, the upper bound gives 
$$
f(w_{k+1}) \le f(w_k) - \frac{1}{L} \|\nabla f(w_k)\|^2 + \frac{L}{2} \|\frac{1}{L} \nabla f(w_k)\|^2 = f(w_k) - \frac{1}{2L} \|\nabla f(w_k)\|^2.
$$
 This shows $f(w_k)$ is non-increasing.
 - Apply the definition of convexity and the update rule to relate progress to an arbitrary vector $u$.
   - *Why:* Using $f(w_k) - f(u) \le \langle \nabla f(w_k), w_k - u \rangle$ and substituting the update, we obtain 
$$
f(w_{k+1}) - f(u) \le L \langle w_k - w_{k+1}, w_{k+1} - u \rangle + \frac{L}{2} \|w_k - w_{k+1}\|^2.
$$

 - Invoke the 3-point identity for Euclidean distances.
   - *Why:* Using the identity $2 \langle c-b, b-a \rangle = \|c-a\|^2 - \|c-b\|^2 - \|b-a\|^2$, the previous inequality simplifies to: 
$$
f(w_{k+1}) - f(u) \le \frac{L}{2} ( \|w_k - u\|^2 - \|w_{k+1} - u\|^2 ).
$$

 - Sum this inequality from $k=0$ to $T-1$ with $u = w^*$.
   - *Why:* On the left, we use $\sum (f(w_{k+1}) - f(w^*)) \ge T(f(w_T) - f(w^*))$ because the sequence is non-increasing. On the right, the sum telescopes to $\frac{L}{2} ( \|w_0 - w^*\|^2 - \|w_T - w^*\|^2 )$.
 - Conclude the sublinear rate.
   - *Why:* Rearranging gives $f(w_T) - f(w^*) \le \frac{L}{2T} \|w_0 - w^*\|^2$, which is $O(1/T)$.
 - For the linear rate under strong convexity, combine the descent step with the lower bound $f(w_k) - f(w^*) \ge \frac{\mu}{2} \|w_k - w^*\|^2$.
   - *Why:* This yields a recursive contraction $\|w_{k+1} - w^*\|^2 \le \rho \|w_k - w^*\|^2$ with $\rho < 1$.
 **Conclusion.** Iterates thus converge linearly toward the unique minimum.

## Links
**Parent:** [[obj_121 - Theorem 22]]

**Prerequisites:**
- [[obj_121 - Theorem 22]]
