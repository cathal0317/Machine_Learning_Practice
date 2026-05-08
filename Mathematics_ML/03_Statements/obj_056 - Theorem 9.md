---
id: obj_056
title: "Theorem 9"
types:
  - theorem
page_start: 26
page_end: 26
parent_id: "obj_051"
children_ids: []
sibling_ids:
  - obj_052
  - obj_053
  - obj_054
  - obj_055
  - obj_057
prerequisites:
  - obj_025
  - obj_052
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Theorem 9

## Conceptual overview
This theorem is the culmination of the subsection, proving that our expected error in learning is controlled by the Rademacher complexity of the function class resulting from composing our loss function with our hypothesis class.

## Why it matters
It allows us to prove that complex models (like deep networks) can still generalize, as long as we can show their Rademacher complexity is small.

## Active recall
> [!question]- How does Theorem 9 bound the estimation error?
> It shows that $\mathbb{E}(R(\hat{h}) - R(\bar{h})) \le 2\mathcal{R}_n(\mathcal{F})$, where $\mathcal{F}$ is the class of functions representing the loss of each hypothesis.

## Exact statement
Let $\mathcal{F} := \{(x, y) \to \ell(h(x), y) ; h \in \mathcal{H}\}$. Then, 
$$
\mathbb{E}(R(\hat{h}) - R(\bar{h})) \le 2\mathcal{R}_n(\mathcal{F}).
$$

## Links
**Parent:** [[obj_051 - 1.6.3 Rademacher complexity]]

**Prerequisites:**
- [[obj_025 - Definition 1]]
- [[obj_052 - Definition 3 (Rademacher complexity)]]
