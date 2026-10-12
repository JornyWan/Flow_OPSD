#!/bin/bash
q="$1"
enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$q")
echo "=== $q"
curl -s -m 60 "https://api.semanticscholar.org/graph/v1/paper/search?query=$enc&limit=15&fields=title,year,externalIds" | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin)
  if 'data' not in d: print('ERR',str(d)[:200]); sys.exit()
  for p in d['data']: print(p.get('year'), (p.get('externalIds') or {}).get('ArXiv'), p['title'])
except Exception as e: print('ERR',e)
"
