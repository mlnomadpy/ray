# Proposition: The IMQ Factor Is a Finite Difference in $b$
**Label:** `prop:imq_findiff` | **Location:** main.tex line 890

## What it says
For every $x, w \in \mathbb{R}^d$, every $b \ge 0$, and every step $h > 0$, the forward second difference in the bias satisfies
$$\frac{k_{ⵟ,b+2h}(w,x) - 2\,k_{ⵟ,b+h}(w,x) + k_{ⵟ,b}(w,x)}{2h^2} = \frac{1}{\|w-x\|^2 + \varepsilon} = h_\varepsilon(w,x)$$
**exactly** — no limit $h \to 0$ is required, the identity holds for every step size, and all three biases $b$, $b+h$, $b+2h$ lie inside the kernel's domain $b \ge 0$. Consequently the family span $\operatorname{span}\{k_{ⵟ,b} : b \ge 0\}$ contains the inverse-multiquadric kernel.

## Why it matters
This is the only "spans IMQ" statement the paper uses, and it keeps that claim entirely self-contained: the IMQ-containment of the ⵟ-family is proved in three lines without importing any result from the action-at-a-distance monograph (which is cited only for historical context and fixed-$b$ universality not needed here). It certifies that the radial factor $h_\varepsilon$ — the completely monotone half of the Schur factorization (eq:schur), the object the entire Bernstein–Widder sampling pipeline targets — is itself recoverable from the kernel family by exact bias arithmetic. The **forward** form is the load-bearing refinement: the centered second difference would require $b \ge h$, while the forward difference keeps every evaluated bias in the legal domain $b \ge 0$, so the identity holds at the boundary $b = 0$ too. Exactness (not asymptotic accuracy) means the reduction carries no truncation error.

## Proof idea
The numerator $(w^\top x + b)^2$ is a quadratic polynomial in $b$, and the second difference of any quadratic is the constant $2h^2$ regardless of the base point: with $t = w^\top x + b$,
$$(t+2h)^2 - 2(t+h)^2 + t^2 = 2h^2.$$
The denominator $\|w-x\|^2 + \varepsilon$ is independent of $b$, so it factors out of the difference; dividing by $2h^2$ leaves exactly $h_\varepsilon(w,x)$. The forward stencil $\{b, b+h, b+2h\}$ keeps all biases $\ge b \ge 0$.

## Connections
**Depends on:** the definition of $k_{ⵟ,b}$ (eq:yat_biased); the elementary fact that a second difference annihilates the quadratic in $b$ up to the constant $2h^2$; the $b$-independence of the radial denominator (the same structural fact exploited by prop:biased_feature).
**Used by:** the introduction's claim that $\operatorname{span}\{k_{ⵟ,b} : b \ge 0\}$ contains the IMQ kernel (sec:intro); Section sec:step-schur (line 121), where it certifies the radial factor as a member of the family span and keeps the construction self-contained; the framing of $k_{ⵟ,b}$ as the flagship Bernstein–Schur instance whose radial factor is IMQ.
**Validated by:** exact algebraic identity; no experiment needed (the analogous identity is verified at machine precision in the companion theory monograph).
