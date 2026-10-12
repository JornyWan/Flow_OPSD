import sys, json, subprocess, urllib.parse, time
n=int(sys.argv[1]); L=int(sys.argv[2])
for q in sys.argv[3:]:
    url="https://huggingface.co/api/papers/search?"+urllib.parse.urlencode({'q':q})
    r=subprocess.run(['curl','-s','-m','40',url],capture_output=True,text=True)
    try: d=json.loads(r.stdout)
    except Exception: print('ERR',r.stdout[:200]); continue
    print(f"=== {q}")
    if not isinstance(d,list): print("  ERRRESP",str(d)[:200]); continue
    for x in d[:n]:
        p=x.get('paper',x)
        t=' '.join(p.get('title','').split()); s=' '.join((p.get('summary') or '').split())
        print(f"  [{p.get('publishedAt','')[:10]}] {p.get('id')} :: {t}")
        if L>0: print(f"      {s[:L]}")
    time.sleep(0.5)
