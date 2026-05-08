---
id: obj_067
title: "Lemma 12 (Sauer-Shelah Lemma)"
types:
  - lemma
page_start: 29
page_end: 29
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_062
  - obj_063
  - obj_064
  - obj_065
  - obj_066
prerequisites:
  - obj_058
  - obj_062
used_in: []
analogous_to: []
same_pattern_as: []
family: "theorem-family"
---

# Lemma 12 (Sauer-Shelah Lemma)

## Conceptual overview
The lemma states that if a hypothesis class has a finite VC dimension $d$, its growth function $\Pi_{\mathcal{H}}(n)$ is bounded by $(n+1)^d$. This means that once the number of points exceeds the VC dimension, the number of distinct behaviors grows polynomially rather than exponentially.

## Why it matters
It is the bridge that turns a finite combinatorial property (VC dimension) into a quantitative bound on Rademacher complexity, and thus on generalization risk.

## Active recall
> [!question]- What is the bound on the expected excess risk provided by Sauer-Shelah?
> $\mathbb{E}(R(\hat{h}) - \inf_{h \in \mathcal{H}} R(h)) \leq 2 \sqrt{\frac{2 VC(\mathcal{H}) \log(n+1)}{n}}$.

## Exact statement
Let $\mathcal{H}$ be a class of finite VC dimension $d$. Then, $\Pi_{\mathcal{H}}(n) \leq (n+1)^d$.

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Prerequisites:**
- [[obj_058 - Lemma 10. [Massart’s Lemma]]]
- [[obj_062 - Definition 4]]
