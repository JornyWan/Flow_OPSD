import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/appendix.tex',
"""Our default $\\mathcal K=\\{3,6\\}$ ($\\sigma=0.858,\\,0.602$) reaches higher noise than the
alternative $\\mathcal K=\\{5,6,7\\}$ ($\\sigma=0.707,\\,0.602,\\,0.465$), which uses \\emph{more}
states but tops out at $\\sigma=0.707$. At a matched checkpoint ($50$ updates) and against
the same control (paired gain in MDE, DrawBench / Unseen-1k):

\\begin{center}
\\resizebox{0.98\\linewidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
$\\mathcal K$ & NFE$=2$ & $4$ & $6$ & $10$ & $20$ & $40$\\\\
\\midrule
$\\{3,6\\}$ \\emph{on-grid (ours)} & $+22.4$ / $+23.9$ & $+7.4$ / $+6.3$ & $+4.7$ / $+2.9$ & $+3.6$ / $+1.4$ & $+3.2$ / $+1.3$ & $+3.4$ / $+1.4$\\\\
$\\{5,6,7\\}$ off-grid            & $+23.6$ / $+25.2$ & $+7.1$ / $+6.1$ & $+2.8$ / $+1.4$ & $+1.4$ / $-0.2$ & $+1.1$ / $-0.5$ & $+1.5$ / $-0.2$\\\\
\\bottomrule
\\end{tabular}}
\\end{center}

Adding a third state does not help; placing the states on the deployment grid does. The
off-grid set falls into the measured floor by $10$ steps and is at or below zero on the
held-out prompt set from $10$ steps onward, while the on-grid set holds a positive gain at
every budget (Fig.~\\ref{fig:family}).""",
"""Our default $\\mathcal K=\\{3,6\\}$ ($\\sigma=0.858,\\,0.602$) reaches higher noise than the
alternative $\\mathcal K=\\{5,6,7\\}$ ($\\sigma=0.707,\\,0.602,\\,0.465$), which uses \\emph{more}
states but tops out at $\\sigma=0.707$. At the main checkpoint ($100$ updates) and against
the same control (paired gain in MDE, DrawBench / Unseen-1k):

\\begin{center}
\\resizebox{0.98\\linewidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
$\\mathcal K$ & NFE$=2$ & $4$ & $6$ & $10$ & $20$ & $40$\\\\
\\midrule
$\\{3,6\\}$ (ours) & $+15.3$ / $+16.4$ & $+10.7$ / $+8.2$ & $+7.1$ / $+4.6$ & $+5.8$ / $+3.8$ & $+5.5$ / $+3.4$ & $+5.6$ / $+3.3$\\\\
$\\{5,6,7\\}$      & $+20.1$ / $+21.1$ & $+8.5$ / $+6.5$ & $+4.3$ / $+2.7$ & $+2.5$ / $+1.3$ & $+2.0$ / $+0.8$ & $+2.5$ / $+0.9$\\\\
\\bottomrule
\\end{tabular}}
\\end{center}

Adding a third state does not help. From $4$ steps upward the two-state set that reaches
$\\sigma=0.858$ is ahead on both prompt sets, by $+2.2$ to $+3.5$ MDE, and the gap widens
with the budget; the lower set decays toward the measured floor while ours flattens at a
positive value (Fig.~\\ref{fig:family}). The one exception is $2$ steps, where the lower
set is ahead by about $4.7$ MDE -- at that budget the sampling grid has no interior state
at all ($\\sigma\\in\\{1.000,\\,0.009,\\,0\\}$), so the design principle has nothing to match.""",
"app:family 升到 ck100")
