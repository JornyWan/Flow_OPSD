import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/method.tex',
"""What the
data supports, however, is the weaker and more useful statement that the set must
\\emph{reach high enough noise}: replacing $\\{3,6\\}$ by the lower, denser
$\\{5,6,7\\}$ ($\\sigma=0.707,\\,0.602,\\,0.465$) costs $2$--$3.5$ MDE at every budget from $4$
to $40$ steps despite using \\emph{more} states (App.~\\ref{app:family}). Grid alignment
alone does not explain the effect: the advantage of $\\{3,6\\}$ is smallest at exactly the
budget it matches.""",
"""What the data supports is
an \\emph{interior optimum} in the noise level the set reaches. Replacing $\\{3,6\\}$ by the
lower, denser $\\{5,6,7\\}$ ($\\sigma$ up to $0.707$) costs $2$--$3.5$ MDE at every budget from
$4$ to $40$ steps despite using \\emph{more} states; replacing it by the \\emph{higher}
$\\{2,3,4\\}$ ($\\sigma$ up to $0.913$) is no better and is worse at the full budget
(App.~\\ref{app:family}). The default sits between them. This matches the measured
reward-per-displacement exchange rate, which collapses from $0.457$ at $\\sigma=0.278$ to
$0.007$ at $\\sigma=0.913$ (Sec.~\\ref{sec:analysis}): too little noise and the supervision
misses the states a few-step sampler visits, too much and the reward gradient has almost
no leverage there. Grid alignment alone does not explain the effect -- the advantage of
$\\{3,6\\}$ is smallest at exactly the budget it matches.""", "method:改成内部极值")
