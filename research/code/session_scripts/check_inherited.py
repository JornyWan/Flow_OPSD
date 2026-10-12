import re
SRC="CA-OPD/reference_papers/arXiv-2608.24646v1/tables/main_results.tex"
src=open(SRC,encoding='utf-8',errors='replace').read()
gen=open("paper_figs/make_table_official.py",encoding='utf-8').read()
rows=re.findall(r'\("([^"]+)",\s*"\\\\?[cx]mark",\s*"([^"]*)",\s*R\(([^)]*)\)\)', gen)
print("  抄录行数: %d" % len(rows))
bad=[]
for nm,upd,vals in rows:
    nums=[v.strip() for v in vals.split(',')]
    hit=False
    for line in src.split(r'\\'):
        if all(re.search(r'(?<![0-9.])'+re.escape(n)+r'(?![0-9])', line) for n in nums):
            hit=True; break
    label=nm.replace('\\quad ','').replace('$+$','+').replace('\\opsd','OPSD').replace('\\','')
    print("  %-24s %-5s %-36s %s" % (label[:24], upd, ",".join(nums), "✅" if hit else "❌"))
    if not hit: bad.append((label,nums))
print("\n  ⇒ %d/%d 命中" % (len(rows)-len(bad), len(rows)))
for label,nums in bad:
    print("  ❌ %s 的值 %s 在原文中找不到同行 —— 需人工核" % (label, nums))
