---
id: obj_072
title: "Definition 6"
types:
  - definition
page_start: 33
page_end: 33
parent_id: "obj_071"
children_ids: []
sibling_ids:
  - obj_074
  - obj_076
  - obj_077
  - obj_080
  - obj_081
  - obj_082
prerequisites: []
used_in:
  - obj_074
  - obj_076
  - obj_080
  - obj_081
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 6

## Conceptual overview
A set is convex if it contains all line segments between any of its points. A function is convex if its domain is a convex set and it satisfies the inequality $f(\lambda v + (1-\lambda)w) \leq \lambda f(v) + (1-\lambda)f(w)$. Strictly convex functions satisfy this with a strict inequality for distinct points.

## Why it matters
Strict convexity is important because it guarantees that the minimizer, if it exists, is unique.

## Active recall
> [!question]- What is the difference between convexity and strict convexity?
> Convexity allows for 'flat' regions (linear segments), while strict convexity requires the function to be strictly 'curved' upwards, ensuring a unique minimum.

## Mental picture
A convex set is like a blob with no 'dents' (no caves). A convex function is like a bowl; if you drop a ball in it, it eventually rolls to the bottom. A concave function is just an upside-down bowl.

## Common confusions
Thinking a function must be differentiable to be convex; $f(x) = |x|$ is convex but not differentiable at 0.

## Links
**Parent:** [[obj_071 - 2.1.1 Convexity]]

**Used in:**
- [[obj_074 - Theorem 13]]
- [[obj_076 - Theorem 14 (Characterization of convexity via differentiability)]]
- [[obj_080 - Proposition 15 (First order optimality condition - unconstrained)]]
- [[obj_081 - Proposition 16 (First order optimality condition - constrained)]]
