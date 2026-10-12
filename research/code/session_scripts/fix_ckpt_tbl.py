import io, re
def sub(p,o,n,tag):
    s=io.open(p,encoding='utf-8').read(); assert s.count(o)==1,"MISS %s (%d)"%(tag,s.count(o))
    io.open(p,'w',encoding='utf-8').write(s.replace(o,n,1)); print("  ✅ "+tag)

# 附录:手写表 → 生成器产物
s=io.open('sections/appendix.tex',encoding='utf-8').read()
m=re.search(r'(The few-step gain of propagation.*?\\end\{center\})', s, re.S)
assert m, "找不到附录 ckpt 表"
new = """The few-step gain of propagation over canonical at $4$ sampling steps, as a function of
training checkpoint (paired, $n=1000$, MDE units). Every entry compares an arm against a
canonical baseline trained on the \\emph{same} hardware and world size at the \\emph{same}
checkpoint; cells without such a control are left blank rather than filled with a
cross-hardware comparison.
\\input{tables/table_ckpt}"""
io.open('sections/appendix.tex','w',encoding='utf-8').write(s.replace(m.group(1), new, 1))
print("  ✅ appendix:手写表换成 \\input{tables/table_ckpt}")

# 正文:修正跨硬件对照得来的 ck50 数值
sub('sections/experiments.tex',
"""The default
$\\mathcal K=\\{3,6\\}$ is measurable at the two checkpoints with a same-hardware control and
is positive at both ($+7.41$ and $+10.70$ on DrawBench at $50$ and $100$ updates;
$+6.34$ and $+8.20$ on Unseen-1k), and it grows with training rather than decaying
(App.~\\ref{app:ckpt}).""",
"""The default
$\\mathcal K=\\{3,6\\}$ is now measurable at four checkpoints against same-hardware controls
and is positive at all of them: $+6.66$, $+5.56$, $+7.52$, $+10.70$ MDE on DrawBench at
$20$, $30$, $50$ and $100$ updates, and $+5.95$, $+4.83$, $+6.06$, $+8.20$ on Unseen-1k.
The gain is already significant at $20$ updates, so it is not an artifact of extended
training (App.~\\ref{app:ckpt}).""", "正文 ckpt 数值(改用同硬件对照)")
