import sys, json, urllib.request, time
for i in sys.argv[1:]:
    try:
        p=json.loads(urllib.request.urlopen(urllib.request.Request('https://huggingface.co/api/papers/'+i,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read())
        print(f"### {i} | {p.get('publishedAt','')[:10]} | {p.get('title')}\n{' '.join((p.get('summary') or '').split())}\n", flush=True)
    except Exception as e:
        print('FAIL',i,e)
    time.sleep(1)
