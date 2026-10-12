import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/experiments.tex',
"""The default
$\\mathcal K=\\{3,6\\}$ is now measurable at four checkpoints against same-hardware controls
and is positive at all of them: $+6.66$, $+5.56$, $+7.52$, $+10.70$ MDE on DrawBench at
$20$, $30$, $50$ and $100$ updates, and $+5.95$, $+4.83$, $+6.06$, $+8.20$ on Unseen-1k.
The gain is already significant at $20$ updates, so it is not an artifact of extended
training (App.~\\ref{app:ckpt}).""",
"""The default
$\\mathcal K=\\{3,6\\}$ is measurable at four checkpoints against same-hardware controls. At
$4$ steps it clears the threshold at every one of them -- $+6.66$, $+5.56$, $+7.52$,
$+10.70$ MDE on DrawBench at $20$, $30$, $50$ and $100$ updates, and $+5.95$, $+4.83$,
$+6.06$, $+8.20$ on Unseen-1k -- so the few-step gain is not an artifact of extended
training. The \\emph{full-budget} gain behaves differently: it grows with training
($+2.39\\!\\to\\!+5.62$ on DrawBench, $+1.38\\!\\to\\!+3.34$ on Unseen-1k) and only clears the
threshold on both prompt sets at $100$ updates. The two budgets therefore mature at
different rates, and the claim that propagation gains at every budget is one we can make
at $100$ updates, not at $20$ (App.~\\ref{app:ckpt}).""", "正文:加满预算随训练成熟的保留")
