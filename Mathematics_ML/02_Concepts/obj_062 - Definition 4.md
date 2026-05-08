---
id: obj_062
title: "Definition 4"
types:
  - definition
page_start: 28
page_end: 28
parent_id: "obj_061"
children_ids: []
sibling_ids:
  - obj_058
  - obj_060
  - obj_063
  - obj_064
  - obj_065
  - obj_066
  - obj_067
prerequisites: []
used_in:
  - obj_067
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Definition 4

## Conceptual overview
Shattering occurs when a hypothesis class can realize every possible binary labeling of a set of points. The growth function $\Pi_{\mathcal{H}}(n)$ represents the maximum number of such distinct labelings possible on any $n$ points. The VC dimension is formally the supremum of $n$ for which shattering is still possible.

## Why it matters
This definition provides a distribution-independent measure of model capacity that directly links to generalization error.

## Active recall
> [!question]- What does it mean for a hypothesis class to 'shatter' a set of points?
> A set of points is shattered if the hypothesis class can achieve every possible binary labeling $(2^n)$ on those points.

> [!question]- Is the VC dimension always finite?
> No, for very complex classes (like the set of all possible classifiers), the VC dimension can be infinite.

## Mental picture
Imagine a set of light switches. Shattering means you can find a 'function' in your class to create every possible combination of on/off positions for those specific switches. The VC dimension is the most switches you can arrange such that you can still create every pattern.

## Common confusions
A common mistake is thinking that shattering must hold for *all* sets of size $n$; in reality, the VC dimension being $n$ only requires that *at least one* set of size $n$ can be shattered.

## Links
**Parent:** [[obj_061 - 1.6.4 Vapnik-Chernovenkis (VC) dimension]]

**Used in:**
- [[obj_067 - Lemma 12 (Sauer-Shelah Lemma)]]
