p='paper_figs/make_table1.py'; s=open(p).read()

old_hdr = '''       "in units of MDE$=0.0033$; win\\\\% in parentheses. Three interventions and two combinations.}",
       "\\\\label{tab:where}", "\\\\resizebox{\\\\linewidth}{!}{%", "\\\\begin{tabular}{lcc cc cc}", "\\\\toprule",
       "& & & \\\\multicolumn{2}{c}{\\\\textbf{DrawBench}} & \\\\multicolumn{2}{c}{\\\\textbf{Unseen-1k}} \\\\\\\\",
       "\\\\cmidrule(lr){4-5}\\\\cmidrule(lr){6-7}",
       "\\\\textbf{Intervention} & \\\\textbf{Cost} & \\\\textbf{Setting} & NFE$=$4 & NFE$=$40 & NFE$=$4 & NFE$=$40 \\\\\\\\",
       "\\\\midrule"]'''
new_hdr = '''       "in units of MDE$=0.0033$; win\\\\% in parentheses. A cell counts as a gain when it clears",
       "$2.5\\\\times$MDE with $t>2$, the threshold calibrated against the measured canonical-versus-canonical",
       "noise floor (Sec.~\\\\ref{sec:exp:setup}); $^\\\\dagger$ marks the $1$--$2.5\\\\times$MDE band, which a single",
       "seed cannot resolve. The last column counts cells that clear the threshold: propagation is the only",
       "intervention positive at every budget on both prompt sets.}",
       "\\\\label{tab:where}", "\\\\resizebox{\\\\linewidth}{!}{%", "\\\\begin{tabular}{lcc cc cc c}", "\\\\toprule",
       "& & & \\\\multicolumn{2}{c}{\\\\textbf{DrawBench}} & \\\\multicolumn{2}{c}{\\\\textbf{Unseen-1k}} & \\\\\\\\",
       "\\\\cmidrule(lr){4-5}\\\\cmidrule(lr){6-7}",
       "\\\\textbf{Intervention} & \\\\textbf{Cost} & \\\\textbf{Setting} & NFE$=$4 & NFE$=$40 & NFE$=$4 & NFE$=$40 & \\\\textbf{Gains} \\\\\\\\",
       "\\\\midrule"]'''
assert s.count(old_hdr)==1, "hdr"
s = s.replace(old_hdr, new_hdr, 1)

old_row = '''for nm, cost, note, vals in rows:
    cs = []
    for v in vals:
        cs.append("--" if v is None else "$%+.2f$ \\\\scriptsize(%.0f\\\\%%)" % (v[0], v[2]))
    out.append("%s & %s & \\\\scriptsize %s & %s \\\\\\\\" % (nm, cost, note, " & ".join(cs)))'''
new_row = '''def _sig(v):   # §4.615:单 seed 要 >=2.5xMDE 且 t>2
    return v is not None and v[0] >= 2.5 and v[1] > 2
for nm, cost, note, vals in rows:
    cs = []
    for v in vals:
        if v is None: cs.append("--"); continue
        dag = "$^\\\\dagger$" if (1.0 <= v[0] < 2.5) else ""
        cs.append("$%+.2f$%s \\\\scriptsize(%.0f\\\\%%)" % (v[0], dag, v[2]))
    ng, nh = sum(_sig(v) for v in vals), sum(v is not None for v in vals)
    gcol = "--" if nh == 0 else ("$\\\\mathbf{%d/%d}$" % (ng, nh) if ng == nh == len(vals) else "$%d/%d$" % (ng, nh))
    out.append("%s & %s & \\\\scriptsize %s & %s & %s \\\\\\\\" % (nm, cost, note, " & ".join(cs), gcol))'''
assert s.count(old_row)==1, "row"
s = s.replace(old_row, new_row, 1)
open(p,'w').write(s); print("  ✅ make_table1.py:加逐格判定列 + 未判带 dagger")
