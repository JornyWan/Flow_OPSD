import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/appendix.tex',
"""Adding a third state does not help, and neither does reaching higher noise. A third
member of the family, $\\mathcal K=\\{2,3,4\\}$ ($\\sigma=0.913,\\,0.858,\\,0.790$), was trained
under the same protocol: against the default it is $+1.7$ MDE at $4$ steps on DrawBench
but $-0.4$ on Unseen-1k, and $-1.8$ and $-2.5$ at the full budget, so it is not an
improvement. Held-out ImageReward disagrees in three of those four cells (it favours the
higher set), which we report but cannot resolve at this training budget. On the reported
composite the three members order as an interior optimum at $\\sigma\\!\\approx\\!0.858$.""",
"""Adding a third state does not help, and neither does reaching higher noise. A third
member, $\\mathcal K=\\{2,3,4\\}$ ($\\sigma$ up to $0.913$), was trained under the same
protocol. At $100$ updates the default beats it in all four cells -- $-1.30$ and $-1.91$
MDE on DrawBench, $-3.24$ and $-2.63$ on Unseen-1k at $4$ and $40$ steps, the last two
clearing the significance threshold against it. Against the canonical baseline the three
members order as $+10.70/+8.20$ (default), $+9.40/+4.96$ (higher), $+8.48/+6.47$ (lower)
at $4$ steps and $+5.62/+3.34$, $+3.70/+0.70$, $+2.51/+0.92$ at $40$: the default is best
in every cell. The supervision set therefore has an \\emph{interior} optimum in the noise
level it reaches, near $\\sigma\\!\\approx\\!0.858$.""",
"app:family 升到 ck100 四格")
