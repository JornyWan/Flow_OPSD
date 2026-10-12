import sys, json, subprocess, urllib.parse, time
for term in sys.argv[1:]:
    url = "https://api2.openreview.net/notes/search?" + urllib.parse.urlencode({'term': term, 'limit': 25, 'source':'forum'})
    for a in range(4):
        r = subprocess.run(['curl','-s','-m','40',url],capture_output=True,text=True)
        try: d=json.loads(r.stdout); break
        except Exception: time.sleep(5); d={}
    print(f"=== {term}")
    seen=set()
    for n in d.get('notes',[]):
        c=n.get('content',{}); t=c.get('title',{}); t=t.get('value',t) if isinstance(t,dict) else t
        if t in seen: continue
        seen.add(t)
        v=c.get('venue',{}); v=v.get('value',v) if isinstance(v,dict) else v
        print(f"  {n.get('id')} [{v}] :: {t}")
    time.sleep(1)
