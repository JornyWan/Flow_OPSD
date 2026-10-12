#!/usr/bin/env python
"""跨节一致性检查:verify_numbers.py 查"数对不对",这个查"同一主张有没有在两处分叉"。
起因(§4.730):额外算力占比在 abstract 写 'a tenth'、正文写 'a fifth',两处都"数值合法",
但互相矛盾 —— 数值校验器结构上抓不到这类错。"""
import re, glob, os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_msp"))
abstract = open("sections/abstract.tex").read()
body = "".join(open(f).read() for f in glob.glob("sections/*.tex") if "abstract" not in f)
bad = 0
# (1) abstract 的每个数都应在正文出现(abstract 不该有独家数字)
vals = sorted({(a or b) for a, b in
               re.findall(r'\$([+-]?\d+\.?\d*)\\?%?\$|\$([+-]?\d+\.\d+)\$', abstract) if (a or b)},
              key=lambda x: -abs(float(x)))
orphan = [v for v in vals if not (("$%s$" % v) in body or ("$%s\\%%$" % v) in body or v in body)]
if orphan:
    bad += len(orphan); print("  🔴 只在 abstract 出现的数: %s" % ", ".join(orphan))
# (2) 文字量词必须一致
for pat in (r'a (?:tenth|fifth|quarter|third|half) of', r'(?:doubl|tripl|quadrupl)\w*'):
    a = set(m.group(0) for m in re.finditer(pat, abstract))
    b = set(m.group(0) for m in re.finditer(pat, body))
    if a - b:
        bad += len(a - b); print("  🔴 abstract 用了正文没有的量词: %s" % sorted(a - b))
print("  %s 跨节一致性:abstract %d 个数、量词全部在正文找到" % ("✓" if not bad else "❌", len(vals)))
sys.exit(1 if bad else 0)
