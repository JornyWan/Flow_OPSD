import json,sys,urllib.request,urllib.parse,time,random
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 15; L=int(sys.argv[3]) if len(sys.argv)>3 else 0
u="https://huggingface.co/api/papers/search?"+urllib.parse.urlencode({"q":q})
d=None
for a in range(8):
    try:
        d=json.load(urllib.request.urlopen(u,timeout=60)); break
    except Exception as e:
        time.sleep(10+random.random()*10)
print("### Q:",q)
if d is None: print("FAILED"); sys.exit()
for x in d[:n]:
    p=x.get('paper',x)
    t=" ".join((p.get('title') or '').split())
    print(p.get('id'), (p.get('publishedAt') or '')[:10], '|', t)
    if L: print("    ", " ".join((p.get('summary') or '').split())[:L])
