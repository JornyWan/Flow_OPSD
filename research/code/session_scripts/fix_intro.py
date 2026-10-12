p='sections/intro.tex'; s=open(p).read()
old="""(ii) Propagation (C) is \\emph{not} a trade. It is positive at every budget we measure,
retaining $37\\%$ of its peak gain at $40$ steps where (A) and (B) retain $3\\%$.
(iii) The two are composable: (B$+$C) attains the largest few-step gain in the study
($+16.23$ MDE at $4$ steps, $92.7\\%$ win rate) while turning (B)'s full-budget loss into a
gain."""
new="""(ii) Propagation (C) is \\emph{not} a trade. It is the only one of the three
interventions that clears our significance threshold in \\emph{all four}
prompt-set~$\\times$~budget cells, retaining $37\\%$ of its peak gain at $40$ steps where
(A) and (B) retain $3\\%$.
(iii) The two are composable, and the composition is a \\emph{choice}: (B$+$C) attains the
largest few-step gain in the study ($+16.23$ MDE at $4$ steps, $92.7\\%$ win rate) and
removes (B)'s full-budget loss, but it does not reach full-budget significance and is
$3.4$ and $1.9$ MDE \\emph{below} pure (C) there. (C) alone remains the configuration to
use when one model must serve every budget."""
assert s.count(old)==1
s=s.replace(old,new,1); open(p,'w').write(s); print("  ✅ intro Findings (ii)/(iii) 已改")
