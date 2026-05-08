---
id: obj_160
title: "Convolutions and Fourier transforms in signal processing"
types:
  - discussion
page_start: 66
page_end: 67
parent_id: "obj_158"
children_ids: []
sibling_ids:
  - obj_156
  - obj_157
  - obj_161
  - obj_162
prerequisites:
  - obj_156
used_in: []
analogous_to: []
same_pattern_as: []
family: "section-family"
---

# Convolutions and Fourier transforms in signal processing

## Conceptual overview
Explains how convolutions are linked to Linear Time-Invariant (LTI) systems. It defines linearity and shift-invariance and notes that any operator satisfying these properties is a convolution with a specific filter.

## Why it matters
Provides the signal-processing justification for why CNNs are designed around the convolution operation.

## Active recall
> [!question]- What are the three defining properties of a Linear Time-Invariant (LTI) operator?
> 1. Linearity: $L(\alpha f) = \alpha L(f)$. 2. Additivity: $L(f + g) = L(f) + L(g)$. 3. Shift-invariance: $L(f(\cdot - \tau)) = (L(f))(\cdot - \tau)$.

> [!question]- What is the 'filter' in the context of an LTI operator?
> The filter $h$ is defined as the operator's response to a Dirac measure, i.e., $h = L(\delta)$.

## Narrative flow
Defines LTI filters, lists their properties, and then provides an informal proof that any LTI operator is necessarily a convolution operator.

## Links
**Parent:** [[obj_158 - 3.4.1 Convolutional neural networks (CNN)]]

**Prerequisites:**
- [[obj_156 - Definition 11 (Convolution)]]
