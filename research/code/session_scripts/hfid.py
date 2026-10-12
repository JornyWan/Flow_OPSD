import sys, json, subprocess, time
for i in sys.argv[1:]:
    r=subprocess.run(['curl','-s','-m','40',f"https://huggingface.co/api/papers/{i}"],capture_output=True,text=True)
    try: p=json.loads(r.stdout)
    except Exception: print('ERR',i,r.stdout[:100]); continue
    print(f"## {i} [{p.get('publishedAt','')[:10]}] {' '.join(p.get('title','').split())}\n{' '.join((p.get('summary') or '').split())}\n")
    time.sleep(0.3)
