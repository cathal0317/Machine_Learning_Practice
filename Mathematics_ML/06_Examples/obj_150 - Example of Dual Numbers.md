---
id: obj_150
title: "Example of Dual Numbers"
types:
  - example
page_start: 62
page_end: 62
parent_id: "obj_149"
children_ids: []
sibling_ids:
  - obj_148
prerequisites:
  - obj_149
used_in: []
analogous_to: []
same_pattern_as: []
family: "technique-family"
---

# Example of Dual Numbers

## Conceptual overview
Provides a step-by-step example of computing a directional derivative by passing dual numbers through the intermediate steps of a function's computation graph.

## Why it matters
Demonstrates the practical validity and simplicity of the dual number approach to automatic differentiation.

## Active recall
> [!question]- What value is computed simultaneously with $f(2,3)$ in this example?
> The directional derivative $(r_1, r_2)\nabla f(2,3)$.

## When to use
When implementing or illustrating forward mode auto-differentiation for specific functions.

## Core pattern
Initializing input nodes as dual numbers $x_i + \varepsilon r_i$ to track sensitivities through a DAG.

## Links
**Parent:** [[obj_149 - Dual numbers]]

**Prerequisites:**
- [[obj_149 - Dual numbers]]
