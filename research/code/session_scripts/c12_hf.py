import sys, json, urllib.request, urllib.parse, time
n=int(sys.argv[1]); L=int(sys.argv[2])
for q in sys.argv[3:]:
    url='https://huggingface.co/api/papers/search?'+urllib.parse.urlencode({'q':q})
    for a in range(4):
        try:
            d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read()); break
        except Exception as e:
            time.sleep(5+5*a); d=None
    print('=== HF:',q)
    for x in (d or [])[:n]:
        p=x.get('paper',x); print(f"- {p.get('id')} | {p.get('publishedAt','')[:10]} | {p.get('title')}"); 
        if L: print('    ',' '.join((p.get('summary') or '').split())[:L])
    print(flush=True); time.sleep(1.5)
