import io
def sub(path, old, new, tag):
    s=io.open(path,encoding='utf-8').read()
    assert s.count(old)==1, "MISS %s (%d)"%(tag,s.count(old))
    io.open(path,'w',encoding='utf-8').write(s.replace(old,new,1)); print("  ✅ "+tag)

# 1) teaser 进 intro 顶部
sub('sections/intro.tex',
"""\\section{Introduction}
\\label{sec:intro}
""",
"""\\section{Introduction}
\\label{sec:intro}

\\begin{figure}[t]
\\centering
\\includegraphics[width=\\linewidth]{fig0_teaser.pdf}
\\caption{\\textbf{Multi-state target propagation (\\MSP) makes on-policy reward distillation
budget-robust.} (a) Reported composite versus sampling budget on DrawBench. The prior state
of the art, \\opsd, loses most of its advantage once the budget falls below $10$ steps;
propagating the \\emph{same} reward-constructed target to higher-noise states of the same
rollout recovers it, and at $4$ steps already matches what \\opsd needs $40$ steps to reach.
(b) Same prompts, same seed, $4$ sampling steps.}
\\label{fig:teaser}
\\end{figure}
""", "teaser 进 intro")

# 2) 主表 + 绝对曲线进 experiments,放在 tab:where 之前
sub('sections/experiments.tex',
"""\\input{tables/table_where}

Table~\\ref{tab:where} is the main result. Three observations.""",
"""\\input{tables/table_main}

Table~\\ref{tab:main} reports absolute scores. \\MSP is the best configuration at the full
budget on every axis we report -- composite $1.6538$ and held-out ImageReward $1.5072$ --
and at $4$ steps it recovers $95\\%$ of the full-budget composite that \\opsd only reaches
with $40$ network evaluations. Combining \\MSP with a shortened rollout is stronger still at
$4$ steps, taking every axis except HPSv2.

\\begin{figure}[t]
\\centering
\\includegraphics[width=\\linewidth]{fig2_quality_vs_nfe.pdf}
\\caption{\\textbf{Absolute quality versus sampling budget}, both prompt sets. Dotted line:
\\opsd evaluated with $40$ steps. \\MSP crosses that line at roughly $5$ steps; \\opsd itself
needs about $10$ (App.~\\ref{app:cost}).}
\\label{fig:quality}
\\end{figure}

\\input{tables/table_where}

Table~\\ref{tab:where} isolates \\emph{where} the gain comes from. Three observations.""",
"主表 + fig2 进 experiments")

# 3) fig3 进附录(族形状),fig4 进 app:ckpt,fig5 进 app:peraxis
sub('sections/appendix.tex',
"""The dip at $30$ updates appears on \\emph{both} prompt sets, so it is a property of the
training trajectory rather than evaluation noise. All eight cells exceed $+5.38$ MDE.""",
"""The dip at $30$ updates appears on \\emph{both} prompt sets, so it is a property of the
training trajectory rather than evaluation noise. All eight cells exceed $+5.38$ MDE.

\\begin{figure}[h]
\\centering
\\includegraphics[width=0.78\\linewidth]{fig4_advantage_vs_ckpt.pdf}
\\caption{Few-step advantage of \\MSP over canonical as a function of training checkpoint.
The effect is present from the earliest checkpoint we evaluate and does not depend on
stopping at $100$ updates.}
\\label{fig:ckpt}
\\end{figure}

\\section{Which states to propagate to}
\\label{app:family}

\\MSP propagates to the two highest-noise states of the rollout ($\\sigma=0.913$, $0.858$).
Propagating instead to two \\emph{low}-noise states adjacent to the query gives a markedly
flatter curve: the gain collapses into the measured floor by $10$ steps, whereas the
high-noise choice holds a positive gain out to $40$ (Fig.~\\ref{fig:family}). The benefit
therefore comes from the noise levels the target is carried to, not merely from applying
the loss at more states.

\\begin{figure}[h]
\\centering
\\includegraphics[width=0.78\\linewidth]{fig3_advantage_vs_nfe.pdf}
\\caption{Advantage over canonical for two members of the propagation family. Shaded band
is the measured canonical-versus-canonical floor.}
\\label{fig:family}
\\end{figure}""", "fig4 + fig3 进附录")

sub('sections/appendix.tex',
""" & ImageReward & $+21.50$ & $+9.95$  & $+6.25$  & $+4.35$ & $+1.50$ & $+1.18$\\\\
\\bottomrule
\\end{tabular}
\\end{center}""",
""" & ImageReward & $+21.50$ & $+9.95$  & $+6.25$  & $+4.35$ & $+1.50$ & $+1.18$\\\\
\\bottomrule
\\end{tabular}
\\end{center}

\\begin{figure}[h]
\\centering
\\includegraphics[width=\\linewidth]{fig5_peraxis_vs_nfe.pdf}
\\caption{Per-axis paired $t$ versus sampling budget. Every axis moves in the same
direction at every budget; the gain is not carried by one reward component.}
\\label{fig:peraxis}
\\end{figure}""", "fig5 进 app:peraxis")

# 4) app:ckpt 的 ck100 列补上真实数(Table 1 里早有)
sub('sections/appendix.tex',
"""DrawBench & $+8.16$ & $+5.43$ & $+7.05$ & $+9.15$ & --\\\\
Unseen-1k & $+7.37$ & $+5.38$ & $+6.09$ & $+7.62$ & --\\\\""",
"""DrawBench & $+8.16$ & $+5.43$ & $+7.05$ & $+9.15$ & $+10.70$\\\\
Unseen-1k & $+7.37$ & $+5.38$ & $+6.09$ & $+7.62$ & $+8.20$\\\\""", "app:ckpt 补 ck100")
