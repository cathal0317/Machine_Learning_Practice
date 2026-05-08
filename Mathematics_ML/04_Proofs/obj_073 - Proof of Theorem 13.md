---
id: obj_073
title: "Proof of Theorem 13"
types:
  - proof
page_start: 33
page_end: 33
parent_id: "obj_074"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_074
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 13

## Conceptual overview
The proof uses the definition of convexity to show that if a better global point existed, points on the line segment leading to it (within any local neighborhood) would also be better than the alleged local minimum, creating a contradiction.

## Why it matters
This result is the primary theoretical justification for using descent algorithms in optimization.

## Active recall
> [!question]- How is uniqueness proven in the strictly convex case?
> By contradiction: if two distinct global minimizers existed, the function value at their midpoint would be strictly lower than the minimum, which is impossible.

## Proves
[[obj_074 - Theorem 13]]

## Proof skeleton
**Goal.** Prove that local minimizers of convex functions are global, and strict convexity implies uniqueness.

**Strategy.** Use contradiction based on the definition of convexity.

- Assume $v$ is a local min but not global, meaning there exists $w$ with $f(w) < f(v)$.

- Define the convex combination $v_\lambda = (1-\lambda)v + \lambda w$. By convexity, $f(v_\lambda) \leq (1-\lambda)f(v) + \lambda f(w) < f(v)$.

- For small $\lambda$, $v_\lambda$ is in the local neighborhood of $v$, contradicting the local optimality of $v$.

- For uniqueness, assume distinct global mins $u, v$. Strict convexity implies $f((u+v)/2) < (1/2)f(u) + (1/2)f(v) = \inf f$.

**Conclusion.** In both cases, the assumptions lead to a contradiction of the definition of a minimizer.

## Full proof with commentary
- We first prove that every local minimizer is global by contradiction. Assume $v$ is a local minimizer but there exists some point $w$ such that $f(w) < f(v)$.
 - Consider the convex combination $v_\lambda = (1-\lambda)v + \lambda w$ for $\lambda \in (0, 1)$.
   - *Why:* By the definition of convexity, $f(v_\lambda) \le (1-\lambda)f(v) + \lambda f(w)$. Since $f(w) < f(v)$, the right-hand side is strictly less than $(1-\lambda)f(v) + \lambda f(v) = f(v)$.
 - Observe that for sufficiently small $\lambda$, the point $v_\lambda$ lies within any given neighborhood of $v$.
   - *Why:* This provides a point in the neighborhood with a strictly lower function value than $v$, which directly contradicts the assumption that $v$ is a local minimizer.
 - To prove uniqueness under strict convexity, assume there are two distinct global minimizers $u$ and $v$.
   - *Why:* By the definition of strict convexity, the value at the midpoint satisfies $f((u+v)/2) < (1/2)f(u) + (1/2)f(v)$. Since $f(u) = f(v) = \inf f$, this implies $f((u+v)/2) < \inf f$, which is impossible.
 **Conclusion.** Therefore, local minimizers of convex functions are global, and they must be unique if the function is strictly convex.

## Links
**Parent:** [[obj_074 - Theorem 13]]

**Prerequisites:**
- [[obj_074 - Theorem 13]]
