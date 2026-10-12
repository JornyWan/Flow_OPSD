# CPU development checks

First integration suite: 58 passed.

The first historical-parity replay stopped on pool1 pair
25ce2d4cd9593140f1e72116. No result file was emitted. Investigation found identical
accept/reject and clause labels, but JSON alphabetical order yielded unknowns
[color,text] while the historical declared order was [text,color]. The shared
decision interface now validates exact declared clause coverage and canonicalizes
to declaration order. Added regression for serialization order and missing clauses.
This does not change the semantic rule or any old result. No image/model call.

Added prospective hardening before final verification: random references must
match the complete verified source log, configuration/source/assets and every
scheduled query; infeasible counts and edited references are refused. Inactive
legacy probability/utility options fail rather than being silently ignored.
