#!/bin/bash
q="$1"
enc=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")
echo "=== $q"
curl -s -m 40 "https://api.alphaxiv.org/search/v2/paper/fast?q=${enc}&includePrivate=false" | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin)
except Exception as e:
  print('ERR',e); sys.exit()
for r in d:
  print(r.get('paperId'),'|',r.get('title'),'|',(r.get('snippet') or '')[:160].replace('\n',' '))
"
