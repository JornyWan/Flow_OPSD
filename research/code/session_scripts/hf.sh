#!/bin/bash
for q in "$@"; do
echo "=== $q"
qq=$(python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")
curl -s -m 30 "https://huggingface.co/api/papers/search?q=$qq" | python -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception as e: print('ERR',e); sys.exit()
for x in d[:15]:
    p=x.get('paper',x); print(p.get('id'),'|',' '.join(p.get('title','').split()),'|',p.get('publishedAt','')[:10])
"
sleep 1; done
