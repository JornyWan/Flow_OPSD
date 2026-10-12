#!/bin/bash
# usage: s.sh "query terms" outfile
q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote_plus(sys.argv[1]))" "$1")
curl -s -m 40 -A "Mozilla/5.0" "https://arxiv.org/search/?query=${q}&searchtype=abstract&abstracts=show&order=-announced_date_first&size=100" -o "$2"
python3 - "$2" <<'PY'
import sys,re,html
t=open(sys.argv[1]).read()
items=t.split('<li class="arxiv-result">')[1:]
print("N=",len(items))
for it in items:
    aid=re.search(r'arxiv.org/abs/([\d.]+)',it)
    title=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',it,re.S)
    date=re.search(r'Submitted</span>\s*(.*?);',it,re.S)
    ab=re.search(r'<span class="abstract-full[^>]*>(.*?)<a',it,re.S)
    clean=lambda s: re.sub(r'\s+',' ',html.unescape(re.sub(r'<.*?>','',s))).strip()
    print("##",aid.group(1) if aid else '?', "|", date.group(1).strip() if date else '?', "|", clean(title.group(1)) if title else '?')
    if ab: print("   ", clean(ab.group(1))[:700])
PY
