#!/bin/bash
q="$1"
enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$q")
echo "=== $q"
curl -s -m 60 "https://export.arxiv.org/api/query?search_query=$enc&max_results=${2:-15}&sortBy=relevance" | python3 -c "
import sys,re
t=sys.stdin.read()
if 'entry' not in t: print('NO/ERR', t[:200]); sys.exit()
for e in t.split('<entry>')[1:]:
  i=re.search(r'<id>(.*?)</id>',e,re.S).group(1).split('/abs/')[-1]
  ti=re.search(r'<title>(.*?)</title>',e,re.S).group(1)
  p=re.search(r'<published>(.*?)</published>',e,re.S).group(1)[:10]
  print(i,p,' '.join(ti.split()))
"
