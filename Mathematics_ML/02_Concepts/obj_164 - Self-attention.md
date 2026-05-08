---
id: obj_164
title: "Self-attention"
types:
  - concept
page_start: 69
page_end: 69
parent_id: "obj_163"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_163
used_in: []
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Self-attention

## Conceptual overview
Self-attention is a mechanism that computes a weighted sum of input tokens where the weights represent a probability vector over the sequence. Each token aggregated is determined by the compatibility of a query with various keys in the context.

## Why it matters
It allows a model to dynamically weigh the importance of different parts of the input data context-dependently, which is crucial for understanding language and complex patterns.

## Active recall
> [!question]- Define the basic self-attention map $Atten_{Q,K,V}(X)(x)$.
> It is defined as $\sum_{i=1}^n \frac{\exp(x^\top Q^\top K x_i)}{\sum_{j=1}^n \exp(x^\top Q^\top K x_j)} V x_i$, where $Q, K, V$ are the query, key, and value matrices.

> [!question]- What is Multi-head self-attention (MHSA)?
> MHSA is the combination of several self-attention maps, allowing the model to attend to information from different representation subspaces at different positions simultaneously.

## Mental picture
Every word in a sentence looks at every other word and asks 'How relevant are you to my current meaning?', then gathers a little bit of information from each according to the answer.

## Common confusions
Thinking self-attention is sequential; it actually computes all token interactions in parallel, making it highly efficient on modern hardware but distinct from recurrent models.

## Links
**Parent:** [[obj_163 - 3.4.2 Transformers]]

**Prerequisites:**
- [[obj_163 - 3.4.2 Transformers]]
