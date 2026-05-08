---
id: obj_116
title: "Definition 10 (Strong convexity and smoothness)"
types:
  - definition
page_start: 49
page_end: 49
parent_id: "obj_118"
children_ids: []
sibling_ids:
  - obj_117
  - obj_120
prerequisites: []
used_in:
  - obj_117
  - obj_121
  - obj_129
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 10 (Strong convexity and smoothness)

## Conceptual overview
Strong convexity and smoothness are two fundamental properties used to bound the curvature of a function. Strong convexity ensures the function has at least a certain minimum curvature (it grows at least as fast as a specific parabola), while smoothness ensures it does not have more than a certain maximum curvature (it grows no faster than a different parabola).

## Why it matters
These definitions allow for the derivation of convergence rates for general differentiable functions, not just quadratics.

## Active recall
> [!question]- What is the mathematical definition of $\mu$-strong convexity?
> A differentiable function $f$ is $\mu$-strongly convex if for all $x, x'$, $\langle \nabla f(x) - \nabla f(x'), x - x' \rangle \ge \mu \|x - x'\|^2$.

> [!question]- What is the mathematical definition of $L$-Lipschitz smoothness?
> A function $f$ is $L$-Lipschitz smooth if its gradient satisfies $\|\nabla f(x) - \nabla f(x')\| \le L \|x - x'\|$ for all $x, x'$.

## Mental picture
Strong convexity acts as a 'safety net' that prevents the function from becoming too flat, ensuring the optimum is 'well-pointed'. Smoothness acts as a 'speed limit' that prevents the gradient from changing too fast, ensuring that a step in the direction of the current gradient is actually useful for a reasonable distance.

## Common confusions
A student might think that any convex function is strongly convex; however, functions like $f(x) = x^4$ are convex but not strongly convex because they become very flat near the origin.

## Links
**Parent:** [[obj_118 - 2.4.2 Strong convexity and smoothness]]

**Used in:**
- [[obj_117 - Proposition 21]]
- [[obj_121 - Theorem 22]]
- [[obj_129 - Theorem 23]]
