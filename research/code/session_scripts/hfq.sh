#!/bin/bash
q="$1"
enc=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$q")
echo "=== $q"
curl -s -m 40 "https://huggingface.co/api/papers/search?q=$enc" | python3 -c "
import sys,json
try:
  d=json.load(sys.stdin)
  for p in d[:15]: print(p['paper']['id'], p['paper']['publishedAt'][:10], p['paper']['title'].replace(chr(10),' '))
except Exception as e: print('ERR',e)
"
