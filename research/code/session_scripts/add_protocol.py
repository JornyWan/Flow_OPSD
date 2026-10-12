import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)

# 1) 换定性图 + caption
sub('sections/experiments.tex',
"""\\includegraphics[width=\\linewidth]{fig6_qualitative.pdf}
\\caption{\\textbf{Same prompt, same seed, same sampler; only the checkpoint changes.}
Prompts are the $N$ highest-scoring DrawBench prompts under the \\emph{canonical} model at
the full budget, so that few-step degradation is the only source of difference; the
selection rule is fixed in advance and applied without manual filtering. A random,
unselected panel is given in App.~\\ref{app:qual}.}""",
"""\\includegraphics[width=\\linewidth]{fig6_qualitative_cfg.pdf}
\\caption{\\textbf{Four sampling steps, same prompt, same seed; only the model changes.}
Prompts are selected in advance by a fixed rule (above-median absolute score under our
model, ranked by paired improvement) and applied without manual filtering; an unselected
panel is in App.~\\ref{app:qual}. These panels are rendered at guidance scale $1.5$ rather
than the CFG-free setting used for every number in this paper, because the CFG-free
protocol is a poor \\emph{display} setting at four steps (App.~\\ref{app:protocol}); the
quantitative protocol is unchanged.}""", "定性图换成 CFG=1.5 版")

# 2) 新增附录章节:评测协议对图像的影响
s=io.open('sections/appendix.tex',encoding='utf-8').read()
s += r"""
\section{The evaluation protocol is not a display protocol}
\label{app:protocol}

Every number in this paper uses the protocol inherited from prior work: $512\times512$,
classifier-free guidance disabled, and the stated number of sampling steps. That protocol
is the right one for \emph{comparison} -- all arms share it and all comparisons are paired
-- but it is a poor setting for \emph{looking at} images, and it also exaggerates one
specific defect.

\paragraph{A frequency measurement.} Reward-trained SD3.5-M checkpoints carry a periodic
texture at the VAE stride ($8$ px) and the patch stride ($16$ px). We quantify it as the
ratio of the spectral magnitude at that period to a local baseline, averaged over the two
image axes; $1.0$ means no periodic structure. The untrained base model scores $1.22$
under the evaluation protocol and $1.03$ once guidance and $40$ steps are restored, so the
structure is not an artifact of the architecture. Reward training raises it: our canonical
reproduction reaches $2.29$ and \MSP $2.57$, while the publicly released single-reward
OPSD checkpoints sit at $1.51$ and $1.56$. Restoring guidance and steps lowers all of them
-- canonical to $1.48$, \MSP to $1.71$, the released checkpoint to $1.22$ -- while the
reported composite \emph{rises} ($1.5460\!\to\!1.6373$ for canonical). The protocol
therefore inflates the defect by roughly $1.5\times$ on top of whatever training produces.

\paragraph{Resolution is not the cause.} At matched guidance and steps, $1024$ and $512$
are indistinguishable ($1.4709$ vs.\ $1.4639$ composite, $1.02$ vs.\ $1.03$ texture).
Removing guidance at $1024$ costs more than halving the resolution does ($1.2380$).

\paragraph{No scalar evaluator sees it.} The texture grows monotonically with training
($1.18$ at $20$ updates to $2.29$ at $100$) while both the reported composite and the
held-out ImageReward improve monotonically over the same range. A frequency measurement is
therefore not redundant with the reward scores; it is the only instrument here that
registers this particular failure.

\paragraph{Choosing a display setting.} Guidance also has an optimum at four steps. Scaling
it from $1.0$ to $4.5$ drives the fraction of pixels clipped to $0$ or $255$ from
$0.4\%$ to $42.8\%$ (canonical) and the composite from $1.5460$ down to $1.3144$. Guidance
$1.5$ maximises the composite ($1.5861$) at $2.6\%$ clipped pixels, and we use it for the
qualitative panels only.
"""
io.open('sections/appendix.tex','w',encoding='utf-8').write(s); print("  ✅ 新增 app:protocol")
