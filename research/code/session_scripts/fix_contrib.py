import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/intro.tex',
"""\\item We give a mechanistic account of why moving the query is a trade and propagating is
not, supported by a measured reward-per-displacement exchange rate across noise levels and
by the height each configuration reaches on the training objective itself
(Sec.~\\ref{sec:analysis}).""",
"""\\item We give a mechanistic account of why moving the query is a trade and propagating is
not, supported by a measured reward-per-displacement exchange rate across noise levels and
by the height each configuration reaches on the training objective itself
(Sec.~\\ref{sec:analysis}).
\\item We characterise the design space the method opens. The noise level the supervision
set reaches has an \\emph{interior} optimum -- stopping lower or reaching higher are both
worse in every cell -- and the gain survives stacking on the strongest baseline in our
study, which rules out the reading that propagation only helps an under-tuned run
(App.~\\ref{app:family}, \\ref{app:stack}).""", "intro:加设计空间贡献")
sub('sections/intro.tex',
"""\\item We report all results against a \\emph{measured} canonical-versus-canonical noise
floor and a calibrated held-out evaluator, and release the calibration
(App.~\\ref{app:floor}, \\ref{app:evalcal}).""",
"""\\item We build on a reproduction that we verify against the published protocol rather than
assume: the untrained base model matches the reported evaluator scores to within $1.6\\%$,
which isolates the evaluation pipeline from any training code
(Sec.~\\ref{sec:exp:repro}). We then report all results against a \\emph{measured}
canonical-versus-canonical noise floor and a calibrated held-out evaluator, and release the
calibration (App.~\\ref{app:floor}, \\ref{app:evalcal}).""", "intro:加复现保真度")
