#!/bin/bash
for id in "$@"; do
curl -s -m 60 -A 'Mozilla/5.0' "https://arxiv.org/abs/$id" | python -c "
import sys,re,html
t=sys.stdin.read()
ti=re.search(r'<h1 class=\"title mathjax\">.*?</span>(.*?)</h1>',t,re.S)
ab=re.search(r'<blockquote class=\"abstract mathjax\">.*?</span>(.*?)</blockquote>',t,re.S)
d=re.search(r'\[Submitted on ([^\]]+)\]',t)
print('##','$id',html.unescape(ti.group(1)).strip() if ti else '?','|',d.group(1) if d else '')
print(' '.join(html.unescape(re.sub(r'<[^>]+>','',ab.group(1))).split()) if ab else '?')
"
sleep 2; done
