import sys, json, urllib.request, urllib.parse
q=sys.argv[1]
url="https://api2.openreview.net/notes/search?"+urllib.parse.urlencode({"term":q,"limit":10,"type":"terms","content":"all","source":"forum"})
d=json.load(urllib.request.urlopen(url,timeout=60))
for n in d.get("notes",[]):
    c=n.get("content",{}); g=lambda k:(c.get(k) or {}).get('value','') if isinstance(c.get(k),dict) else (c.get(k) or '')
    t=g('title')
    if t and q.lower()[:25] in t.lower():
        print('###',t,'|',n.get('domain')); print(g('abstract')[:1800]); print(); break
