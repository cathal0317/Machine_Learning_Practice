---
id: obj_119
title: "Proof of Proposition 21"
types:
  - proof
page_start: 49
page_end: 50
parent_id: "obj_117"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_117
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Proposition 21

## Conceptual overview
The proof establishes quadratic bounds by integrating the directional derivative of the function along a straight line path between two points. By substituting the bounds on the gradient difference provided by strong convexity or smoothness into this integral, the respective quadratic envelopes are derived.

## Why it matters
This proof validates the geometric intuition that curvature bounds can be transformed into value bounds, a key step in convergence proofs.

## Active recall
> [!question]- How is the fundamental theorem of calculus used in the proof of Proposition 21?
> It is used to express the difference $f(y) - f(x)$ as an integral of the directional derivative $h'(t) = \langle \nabla f(x + t(y - x)), y - x \rangle$ from $t=0$ to $t=1$.

> [!question]- What is the key substitution made in the smoothness part of the proof?
> The term $\langle \nabla f(x + t(y-x)) - \nabla f(x), y-x \rangle$ is bounded by $tL\|y-x\|^2$ using Cauchy-Schwarz and the Lipschitz smoothness property.

## Proves
[[obj_117 - Proposition 21]]

## Proof skeleton
**Goal.** Derive quadratic upper and lower bounds for functions satisfying smoothness and strong convexity.

**Strategy.** Define a univariate function $h(t) = f(x + t(y-x))$ and use the fundamental theorem of calculus.

- Write $f(y) - f(x) = \int_0^1 h'(t) dt = \langle \nabla f(x), y-x \rangle + \int_0^1 \langle \nabla f(x + t(y-x)) - \nabla f(x), y-x \rangle dt$.

- To prove the lower bound, apply the $\mu$-strong convexity condition $\langle \nabla f(x + t(y-x)) - \nabla f(x), t(y-x) \rangle \ge \mu t^2\|y-x\|^2$.

- Integrate the resulting term $\int_0^1 \mu t \|y-x\|^2 dt = \frac{\mu}{2}\|y-x\|^2$.

- To prove the upper bound, apply Cauchy-Schwarz to the integral term and use the $L$-smoothness property $\|\nabla f(x + t(y-x)) - \nabla f(x)\| \le tL\|y-x\|$.

- Integrate the resulting term $\int_0^1 t L \|y-x\|^2 dt = \frac{L}{2}\|y-x\|^2$.

**Conclusion.** The inequalities yield the respective quadratic bounds.

## Full proof with commentary
- To establish quadratic bounds, we define a auxiliary univariate function $h(t) = f(x + t(y-x))$ representing the values of $f$ along the line segment between $x$ and $y$.
   - *Why:* The function endpoints are $h(0) = f(x)$ and $h(1) = f(y)$. Its derivative is the directional derivative $h'(t) = \langle \nabla f(x + t(y-x)), y-x \rangle$.
 - Use the fundamental theorem of calculus to relate the endpoints.
   - *Why:* 
$$
f(y) - f(x) = h(1) - h(0) = \int_0^1 h'(t) dt = \int_0^1 \langle \nabla f(x + t(y-x)), y-x \rangle dt.
$$
 We can rewrite this by adding and subtracting $\langle \nabla f(x), y-x \rangle$: 
$$
f(y) - f(x) = \langle \nabla f(x), y-x \rangle + \int_0^1 \langle \nabla f(x + t(y-x)) - \nabla f(x), y-x \rangle dt.
$$

 - To prove the lower bound for $\mu$-strong convexity, bound the integrand from below.
   - *Why:* The definition $(C_\mu)$ implies $\langle \nabla f(x+t(y-x)) - \nabla f(x), t(y-x) \rangle \ge \mu \|t(y-x)\|^2$. Dividing by $t$, we get $\langle \nabla f(x+t(y-x)) - \nabla f(x), y-x \rangle \ge \mu t \|y-x\|^2$.
 - Integrate the lower bound over the interval.
   - *Why:* $\int_0^1 \mu t \|y-x\|^2 dt = \frac{\mu}{2} \|y-x\|^2$. Adding this to the linear Taylor approximation yields Equation 2.12.
 - To prove the upper bound for $L$-smoothness, bound the integrand from above using Cauchy-Schwarz.
   - *Why:* $\langle \dots \rangle \le \|\nabla f(x+t(y-x)) - \nabla f(x)\| \cdot \|y-x\| \le tL \|y-x\|^2$ by the Lipschitz gradient assumption. Integrating $tL \|y-x\|^2$ from 0 to 1 yields $\frac{L}{2}\|y-x\|^2$.
 **Conclusion.** Summing the linear and quadratic terms gives the global upper quadratic envelope.

## Links
**Parent:** [[obj_117 - Proposition 21]]

**Prerequisites:**
- [[obj_117 - Proposition 21]]
