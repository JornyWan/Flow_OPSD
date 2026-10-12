import sys
p='sections/abstract.tex'; s=open(p).read()
old_a="""measurement error -- and that both are a \\emph{trade}: large gains below $10$ sampling
steps, but a small and significant \\emph{loss} at the full budget. Propagation (C) is not
a trade: it gains at every budget we measure. Combining (B) and (C) gives the strongest
few-step model."""
new_a="""measurement error -- and that both are a \\emph{trade}: large gains below $10$ sampling
steps, but a small and significant \\emph{loss} at the full budget. Propagation (C) is not
a trade, and it is the \\emph{only} one of the three that is significant in all four
prompt-set $\\times$ budget cells we evaluate: it gains at every budget we measure."""
old_b="""On SD3.5-Medium under a matched $100$-update budget, propagation improves the reported
composite by $+5.62$ and $+3.34$ MDE at $40$ steps and by $+10.70$ and $+8.20$ MDE at
$4$ steps on two prompt sets, at $1.20\\times$ training cost and with \\emph{no} additional
reward evaluations; the (B$+$C) combination reaches $+16.23$ MDE at $4$ steps with a
$92.7\\%$ per-image win rate. All effects replicate across two random seeds and are
calibrated against a measured canonical-versus-canonical noise floor."""
new_b="""On SD3.5-Medium under a matched $100$-update budget, propagation improves the reported
composite by $+5.62$ and $+3.34$ MDE at $40$ steps and by $+10.70$ and $+8.20$ MDE at
$4$ steps on two prompt sets, at $1.20\\times$ training cost and with \\emph{no} additional
reward evaluations. Propagation also composes with (B), and that combination is the
strongest few-step configuration we measure ($+16.23$ MDE at $4$ steps, $92.7\\%$ per-image
win rate) -- but it inherits (B)'s trade and falls back to parity at the full budget, so we
report it as a deployment option rather than a strictly better method. All effects
replicate across two random seeds and are calibrated against a measured
canonical-versus-canonical noise floor."""
for o,n in ((old_a,new_a),(old_b,new_b)):
    assert s.count(o)==1, ("miss",s.count(o))
    s=s.replace(o,n,1)
open(p,'w').write(s); print("  ✅ abstract:C 升为唯一四格显著;B+C 降为 deployment option")
