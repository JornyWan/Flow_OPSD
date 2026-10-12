import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)

sub('sections/method.tex',
"""Throughout, our default is $\\mathcal K=\\{3,6\\}$, i.e.\\ $\\sigma=0.858$ and $0.602$. These
are not tuned values: they are exactly the two interior states of the \\emph{$4$-step
deployment grid} $\\sigma\\in\\{1.000,\\,0.858,\\,0.602,\\,0.009\\}$, expressed on the $10$-step
training grid. The canonical query $\\sigma^\\star\\!=\\!0.278$ lies on neither -- a model
trained by \\opsd receives supervision at a state that a $4$-step sampler never visits.
\\MSP closes that gap by construction: \\emph{supervise the states the deployment sampler
will actually traverse}. App.~\\ref{app:family} shows this is what matters, not the number
of states: a three-state set placed off the deployment grid decays into the noise floor
while our two-state on-grid set holds its gain out to the full budget.""",
"""Throughout, our default is $\\mathcal K=\\{3,6\\}$, i.e.\\ $\\sigma=0.858$ and $0.602$. These
are not tuned values. They are the two interior states of the $4$-step sampling grid
$\\sigma\\in\\{1.000,\\,0.858,\\,0.602,\\,0.009\\}$ expressed on the $10$-step training grid, and
we picked them for that reason; note that the canonical query $\\sigma^\\star\\!=\\!0.278$ lies
on neither grid, so \\opsd supervises a state a $4$-step sampler never visits. What the
data supports, however, is the weaker and more useful statement that the set must
\\emph{reach high enough noise}: replacing $\\{3,6\\}$ by the lower, denser
$\\{5,6,7\\}$ ($\\sigma=0.707,\\,0.602,\\,0.465$) costs $2$--$3.5$ MDE at every budget from $4$
to $40$ steps despite using \\emph{more} states (App.~\\ref{app:family}). Grid alignment
alone does not explain the effect: the advantage of $\\{3,6\\}$ is smallest at exactly the
budget it matches.""", "method:去掉过度主张")

sub('sections/appendix.tex',
"""Our default $\\mathcal K=\\{3,6\\}$ ($\\sigma=0.858,\\,0.602$) is the $4$-step deployment grid
expressed on the training grid. The alternative $\\mathcal K=\\{5,6,7\\}$ ($\\sigma=0.707,\\,
0.602,\\,0.465$) uses \\emph{more} states but places them off that grid. At a matched
checkpoint ($50$ updates) and against the same control, the two are indistinguishable at
$2$--$4$ steps and then separate (paired gain in MDE, DrawBench / Unseen-1k):""",
"""Our default $\\mathcal K=\\{3,6\\}$ ($\\sigma=0.858,\\,0.602$) reaches higher noise than the
alternative $\\mathcal K=\\{5,6,7\\}$ ($\\sigma=0.707,\\,0.602,\\,0.465$), which uses \\emph{more}
states but tops out at $\\sigma=0.707$. At a matched checkpoint ($50$ updates) and against
the same control (paired gain in MDE, DrawBench / Unseen-1k):""", "app:family 前言")
