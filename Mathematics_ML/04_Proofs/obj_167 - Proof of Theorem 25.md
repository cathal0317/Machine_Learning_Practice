---
id: obj_167
title: "Proof of Theorem 25"
types:
  - proof
page_start: 72
page_end: 72
parent_id: "obj_166"
children_ids: []
sibling_ids: []
prerequisites:
  - obj_166
used_in: []
analogous_to: []
same_pattern_as: []
family: "proof-family"
---

# Proof of Theorem 25

## Conceptual overview
The proof uses calculus of variations and the properties of probability densities to evaluate the GAN value function at its optimal discriminator.

## Why it matters
It establishes the link between adversarial training and standard statistical distance measures (like the Jensen-Shannon divergence, implicitly).

## Active recall
> [!question]- Why is the KL divergence used in the proof of Theorem 25?
> Because KL divergence is non-negative and zero if and only if the distributions are equal, it provides a tool to establish the global lower bound of the objective.

## Proves
[[obj_166 - Theorem 25]]

## Proof skeleton
**Goal.** Show $V(D^*, G) \geqslant -\log(4)$.

**Strategy.** Substitute the expression for $D^*$ into $V(D, G)$ and use properties of logs.

- Write $V(D^*, G) = \int \log(\frac{\rho_X}{\rho_X+\rho_G})\rho_X + \int \log(\frac{\rho_G}{\rho_X+\rho_G})\rho_G$.

- Introduce factors of 2: $\log(\frac{\rho_X}{\rho_X+\rho_G}) = \log(\frac{\rho_X}{(
ho_X+\rho_G)/2}) - \log(2)$.

- Re-express the integrals as $KL(\rho_X || \frac{\rho_X+\rho_G}{2}) + KL(\rho_G || \frac{\rho_X+\rho_G}{2}) - \log(4)$.

**Conclusion.** Since KL is non-negative, the value is minimized at $-\log(4)$ when both KL terms are zero (i.e., $\rho_X = \rho_G$).

## Full proof with commentary
- We evaluate the value function $V(D, G)$ at the optimal discriminator $D^*$ to find the theoretical lower bound.
   - *Why:* Substituting $D^*(x) = \frac{\rho_X(x)}{\rho_X(x) + \rho_G(x)}$ into the integral form of the value function, we obtain 
$$
V(D^*, G) = \int \log\left(\frac{\rho_X}{\rho_X+\rho_G}\right)\rho_X dx + \int \log\left(\frac{\rho_G}{\rho_X+\rho_G}\right)\rho_G dx.
$$

 - Manipulate the logarithms to introduce a factor of 2 in the denominators, facilitating the identification of probability averages.
   - *Why:* We use the identity $\log(a/(a+b)) = \log(a / ((a+b)/2)) - \log(2)$. Applying this to both integral terms results in 
$$
V(D^*, G) = \int \log\left(\frac{\rho_X}{(\rho_X+\rho_G)/2}\right)\rho_X dx - \log(2) + \int \log\left(\frac{\rho_G}{(\rho_X+\rho_G)/2}\right)\rho_G dx - \log(2).
$$

 - Recognize the integral terms as Kullback-Leibler (KL) divergences between the individual class distributions and their mixture.
   - *Why:* The expression simplifies to $KL(\rho_X \mid \frac{\rho_X+\rho_G}{2}) + KL(\rho_G \mid \frac{\rho_X+\rho_G}{2}) - \log(4)$.
 - Determine the minimum value based on the non-negativity of KL divergence.
   - *Why:* Since $KL(P \mid Q) \ge 0$ for any distributions, and equals zero if and only if $P = Q$, the sum of the two KL terms is minimized at zero when $\rho_X = \rho_G$.
 **Conclusion.** Thus, the absolute minimum of the value function is $-\log(4)$, which is achieved if and only if the generator perfectly reproduces the data distribution.

## Links
**Parent:** [[obj_166 - Theorem 25]]

**Prerequisites:**
- [[obj_166 - Theorem 25]]
