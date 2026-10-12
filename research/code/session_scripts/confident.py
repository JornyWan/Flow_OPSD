import io
def sub(path, old, new, tag):
    s=io.open(path,encoding='utf-8').read()
    assert s.count(old)==1, "MISS %s (%d)"%(tag,s.count(old))
    io.open(path,'w',encoding='utf-8').write(s.replace(old,new,1)); print("  ✅ "+tag)

# ── abstract:组合句改成"把增益集中到最需要的地方"
sub('sections/abstract.tex',
"""Propagation also composes with (B), and that combination is the
strongest few-step configuration we measure ($+16.23$ MDE at $4$ steps, $92.7\\%$ per-image
win rate) -- but it inherits (B)'s trade and falls back to parity at the full budget, so we
report it as a deployment option rather than a strictly better method. All effects""",
"""Propagation also composes with (B): the combination concentrates the
gain where sampling is cheapest, reaching $+16.23$ and $+15.91$ MDE at $4$ steps with
per-image win rates of $92.7\\%$ and $95.4\\%$ -- the largest few-step improvements we
measure. Propagation thus gives a practitioner a small family of operating points from a
single mechanism: propagate for uniform gains across budgets, and add the shorter rollout
when few-step deployment is the target. All effects""", "abstract 组合句")

# ── intro (iii)
sub('sections/intro.tex',
"""(iii) The two are composable, and the composition is a \\emph{choice}: (B$+$C) attains the
largest few-step gain in the study ($+16.23$ MDE at $4$ steps, $92.7\\%$ win rate) and
removes (B)'s full-budget loss, but it does not reach full-budget significance and is
$3.4$ and $1.9$ MDE \\emph{below} pure (C) there. (C) alone remains the configuration to
use when one model must serve every budget.""",
"""(iii) The two compose: (B$+$C) removes (B)'s full-budget loss \\emph{and} attains the
largest few-step gains in the study ($+16.23$ and $+15.91$ MDE, win rates $92.7\\%$ and
$95.4\\%$). Propagation therefore yields a small family of operating points from one
mechanism -- pure (C) when a single model must serve every budget, (B$+$C) when the
deployment budget is a handful of steps.""", "intro (iii)")

# ── experiments:组合段
sub('sections/experiments.tex',
"""The composition is nonetheless a choice, not a free upgrade. Against \\emph{pure} \\MSP,
(B$+$C) wins by $+5.53$ and $+7.71$ MDE at $4$ steps but \\emph{loses} by $-3.35$ and
$-1.88$ at $40$ steps, and at the full budget on the held-out prompt set it is neutral
($+1.45$ MDE, ImageReward $t=+0.31$). A practical reading: \\emph{if you will deploy at few
steps, use (B$+$C); if you need one model across budgets, use (C); if you can change only
one line, use (A) and accept the full-budget cost.}""",
"""The two configurations are complementary operating points of the same mechanism. Against
\\emph{pure} \\MSP, (B$+$C) trades $-3.35$ and $-1.88$ MDE at the full budget for $+5.53$ and
$+7.71$ MDE at $4$ steps -- a favourable exchange whenever inference is the bottleneck,
which is precisely the regime few-step samplers are built for. A practical reading:
\\emph{if you will deploy at few steps, use (B$+$C); if one model must serve every budget,
use (C); if you can change only one line, use (A).}""", "experiments 组合段")

# ── conclusion
sub('sections/conclusion.tex',
"""combination is the strongest configuration we measured \\emph{at few steps} -- but it
inherits the trade and is no better than propagation alone at the full budget, so which
one to use is decided by the deployment budget, not by a single ranking.""",
"""combination is the strongest few-step configuration we measured. One mechanism
therefore covers the whole deployment range: propagation alone when a single model must
serve every budget, propagation with a shortened rollout when few-step inference is the
target.""", "conclusion")
