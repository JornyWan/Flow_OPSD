import sys, json, subprocess, urllib.parse, time
# print abstracts of top-k hits
k=int(sys.argv[1])
for term in sys.argv[2:]:
    url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode({'term': term, 'limit': 10, 'source':'forum'})
    r = subprocess.run(['curl','-s','-m','40',url],capture_output=True,text=True)
    try: d=json.loads(r.stdout)
    except Exception: print('ERR', r.stdout[:200]); continue
    print(f"=== {term}")
    seen=set(); c0=0
    for n in d.get('notes',[]):
        c=n.get('content',{})
        g=lambda key:(c.get(key,{}).get('value') if isinstance(c.get(key),dict) else c.get(key))
        t=g('title')
        if t in seen: continue
        seen.add(t); c0+=1
        if c0>k: break
        print(f"## {t} [{g('venue')}]\n{(g('abstract') or '')[:1800]}\n")
    time.sleep(1)
