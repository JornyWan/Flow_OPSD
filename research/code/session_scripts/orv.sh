#!/bin/bash
for q in "$@"; do
echo "=== $q"
qq=$(python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")
curl -s -m 40 "https://api2.openreview.net/notes/search?term=$qq&type=terms&content=all&source=all&limit=20" | python -c "
import sys,json
try: d=json.load(sys.stdin)
except Exception as e: print('ERR',e); sys.exit()
if 'notes' not in d: print(str(d)[:300]); sys.exit()
for n in d['notes']:
    c=n.get('content',{}); t=c.get('title',{}); t=t.get('value',t) if isinstance(t,dict) else t
    v=c.get('venue',{}); v=v.get('value',v) if isinstance(v,dict) else v
    print(n.get('forum'),'|',t,'|',v)
"
sleep 2; done
