import sys, json, urllib.request, urllib.parse, time
def s(term, n=15, abslen=300):
    url='https://api2.openreview.net/notes/search?'+urllib.parse.urlencode({'term':term,'limit':n,'type':'terms','content':'all','source':'forum'})
    for a in range(5):
        try:
            d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read()); break
        except Exception as e:
            err=e; time.sleep(5+5*a)
    else:
        print('FAIL',term,err); return
    print('=== OR:',term, 'n=',d.get('count'))
    for no in d.get('notes',[]):
        c=no.get('content',{})
        g=lambda k: (c.get(k,{}) or {}).get('value','') if isinstance(c.get(k),dict) else c.get(k,'')
        t=g('title'); v=g('venue') or no.get('invitation','') or ''
        ab=' '.join(str(g('abstract')).split())[:abslen]
        cd=time.strftime('%Y-%m-%d',time.gmtime((no.get('cdate') or 0)/1000))
        print(f'- [{cd}] {t} | {v} | https://openreview.net/forum?id={no.get("forum")}\n    {ab}')
    print(flush=True)
for t in sys.argv[1:]:
    s(t); time.sleep(3)
