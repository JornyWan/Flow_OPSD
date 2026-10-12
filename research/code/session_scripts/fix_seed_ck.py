import io
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)
sub('sections/experiments.tex',
"""\\paragraph{Seeds.} A second random seed, on different hardware and against its own
matched canonical baseline, reproduces all four cells with slightly larger magnitude:
$+8.45$ vs.\\ $+7.52$ and $+4.04$ vs.\\ $+3.19$ (DrawBench, $4$ and $40$ steps),
$+6.50$ vs.\\ $+6.06$ and $+1.69$ vs.\\ $+1.22$ (Unseen-1k). The largest seed-to-seed
difference is $0.93$ MDE, below the measured noise floor.""",
"""\\paragraph{Seeds.} A second random seed, run on different hardware and at a different
world size, and compared against \\emph{its own} matched canonical baseline, reproduces the
effect. At $50$ updates all four cells replicate with slightly larger magnitude --
$+8.45$ vs.\\ $+7.52$ and $+4.04$ vs.\\ $+3.19$ (DrawBench, $4$ and $40$ steps),
$+6.50$ vs.\\ $+6.06$ and $+1.69$ vs.\\ $+1.22$ (Unseen-1k) -- with a largest seed-to-seed
difference of $0.93$ MDE, below the measured floor. At $100$ updates the two DrawBench
cells give $+9.65$ and $+3.90$ against $+10.70$ and $+5.62$: the same sign, both still
clearing the threshold, with differences of $1.05$ and $1.72$ MDE. The second of those
exceeds the same-configuration floor, but these two runs differ in hardware and world size
as well as in seed, so we read it as agreement in sign and significance rather than in
magnitude.""", "Seeds 段")
