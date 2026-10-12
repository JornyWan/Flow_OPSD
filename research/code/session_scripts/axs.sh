#!/bin/bash
# arxiv html search: $1 query terms
q=$(python -c "import urllib.parse,sys;print(urllib.parse.quote_plus(sys.argv[1]))" "$1")
curl -s -m 60 -A 'Mozilla/5.0' "https://arxiv.org/search/?query=$q&searchtype=all&abstracts=hide&order=-announced_date_first&size=50" > /tmp/axs_$$.html
python - /tmp/axs_$$.html <<'PY'
import re,sys,html
t=open(sys.argv[1]).read()
items=t.split('<li class="arxiv-result">')[1:]
for it in items:
    m=re.search(r'arxiv.org/abs/([0-9.]+)',it); ti=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',it,re.S)
    d=re.search(r'Submitted</span>\s*([^;<]+)',it)
    print(m.group(1) if m else '?', '|', html.unescape(re.sub(r'<[^>]+>','',ti.group(1))).strip().replace('\n',' ') if ti else '?', '|', d.group(1).strip() if d else '')
PY
rm -f /tmp/axs_$$.html
