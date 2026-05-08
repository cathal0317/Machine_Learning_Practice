---
id: obj_003
title: "Regression vs classification"
types:
  - definition
page_start: 5
page_end: 5
parent_id: "obj_005"
children_ids: []
sibling_ids:
  - obj_004
  - obj_007
  - obj_008
  - obj_009
prerequisites: []
used_in:
  - obj_005
analogous_to: []
same_pattern_as: []
family: "concept-family"
---

# Regression vs classification

## Conceptual overview
This definition categorizes supervised learning tasks based on the nature of the response variable $Y$. Regression deals with quantitative, numerical outputs (e.g., house prices), while classification deals with categorical, qualitative outputs (e.g., spam indicators or class labels). This distinction is fundamental because it dictates the choice of loss function and the structure of the mathematical model.

## Why it matters
Identifying the problem type is the first step in statistical learning, as it determines whether the goal is to predict a continuous value or a discrete label.

## Active recall
> [!question]- What is the primary difference between a regression problem and a classification problem?
> In a regression problem, the output variable $Y$ is quantitative (numerical), whereas in a classification problem, the output variable is categorical (e.g., $\{ -1, 1 \}$).

> [!question]- Provide an example of a regression task from the notes.
> Predicting the price of a house based on the number of bedrooms and other features is a regression task.

## Mental picture
Imagine a number line for regression, where the target can be any point; for classification, imagine a set of distinct buckets into which every input must be sorted.

## Common confusions
Students sometimes confuse the nature of the output with the nature of the input features; the regression/classification distinction depends solely on the response variable $Y$.

## Links
**Parent:** [[obj_005 - 1.1 Classification and regression]]

**Used in:**
- [[obj_005 - 1.1 Classification and regression]]
