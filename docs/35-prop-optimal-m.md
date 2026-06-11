# Proposition: Optimal Sketch Size
**Label:** `prop:optimal_m` | **Location:** main.tex line 874

## What it says

At a fixed total feature budget, the sketch size $m$ trades against the radial-draw count $D$, and the optimum is interior with a closed form.

**Setup.** Fix the feature budget $M = D(m + d + 1)$ (the quadratic-only deployed map costs $m + d + 1$ coordinates per radial draw). To leading order in the two sampling counts, the deployed variance of `prop:ts_variance` is

$$V(m) = \underbrace{\frac{A\,(m + d + 1)}{M}}_{\text{radial},\ A = p^2 V_{\mathrm{rad}}} + \underbrace{\frac{B}{m}}_{\text{sketch},\ B = h^2 C\|x\|^4\|w\|^4} + O\Bigl(\frac{1}{M}\Bigr),$$

with $A$ the radial Monte-Carlo constant ($V_{\mathrm{rad}} = \mathrm{Var}[\widehat h_1]$, the one-draw radial variance) and $B$ the degree-2 TensorSketch constant.

**Conclusion.** $V$ is minimized at the interior point

$$m^\star = \sqrt{\frac{B\,M}{A}}, \qquad V(m^\star) = \frac{2\sqrt{AB}}{\sqrt{M}} + \frac{A(d+1)}{M}.$$

The optimal sketch size grows as $m^\star \propto \sqrt{M}$ with the budget and as $\sqrt{B/A}$ with the sketch-to-radial constant ratio; its $d$-dependence enters only through that ratio, hence is slow.

## Why it matters

This turns the radial-vs-sketch split of `thm:ts_opnorm` and `prop:ts_variance` into an actionable allocation rule: given a feature budget $M$, set $m^\star = \sqrt{BM/A}$ rather than guessing. It explains why the empirical optimum in Table tab:dm is interior — neither "all radial draws, tiny sketch" nor "huge sketch, few draws" is right, because the radial term pays linearly per sketch coordinate through $D = M/(m+d+1)$ while the sketch term decays as $1/m$. The $\sqrt{M}$ growth law and the slow $d$-dependence are both falsifiable predictions, and both land (see Validated by). It also encodes the goal-dependence noted in Section sec:exp_ts: the rule optimizes per-entry Gram variance; downstream KRR instead favors smaller $m$ (more radial draws), so the operating point depends on whether Gram fidelity or prediction is the target.

## Proof idea

One-variable calculus on the two-term law. Substitute the budget constraint $D = M/(m + d + 1)$ into the radial term $A/D$ of `prop:ts_variance`, giving $A(m+d+1)/M$; the sketch term $B/m$ is budget-independent; the interaction term is $O(1/M)$ and absorbed. Then

$$\frac{dV}{dm} = \frac{A}{M} - \frac{B}{m^2} = 0 \quad\Longrightarrow\quad m^\star = \sqrt{\frac{BM}{A}},$$

and substituting back gives $V(m^\star) = 2\sqrt{AB}/\sqrt{M} + A(d+1)/M$.

## Connections

**Depends on:** `prop:ts_variance` (the three-term decomposition whose leading terms form $V(m)$), the budget constraint $M = D(m+d+1)$ of the quadratic-only deployed map (Step 5), the degree-2 TensorSketch variance constant.
**Used by:** Section sec:exp_ts and Table tab:dm (the accuracy–cost trade at fixed $M$), the deployment guidance of the Discussion (RAY as the default scalable map with a principled $m$).
**Validated by:** `dm_tradeoff.py` (Table tab:dm: off-sphere $d \in \{16, 64, 256\}$, $m \in \{16,\dots,512\}$, $M \in \{4096, 8192, 16384\}$ — the optimum is interior) and `sketch_size_rule.py` (fitting $A, B$ to the deployed-allocation data: the two-term law explains $88$–$90\%$ of squared-error variance ($R^2$), the predicted $m^\star = \sqrt{BM/A}$ lands at or within one grid step of the empirical error minimum, and $\sqrt{B/A}$ rises only from $0.41$ to $0.58$ as $d$ goes $16 \to 256$ — the slow $d$-dependence the rule predicts).
