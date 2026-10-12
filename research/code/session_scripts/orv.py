import sys, json, urllib.request, urllib.parse
q=sys.argv[1]
url="https://api2.openreview.net/notes/search?"+urllib.parse.urlencode({"term":q,"limit":50,"type":"terms","content":"all","source":"forum"})
d=json.load(urllib.request.urlopen(url,timeout=60))
seen=set()
for n in d.get("notes",[]):
    c=n.get("content",{})
    t=(c.get("title") or {}).get("value") if isinstance(c.get("title"),dict) else c.get("title")
    if not t or t in seen: continue
    seen.add(t)
    v=n.get("domain") or n.get("invitation") or ""
    print(f"- {t} | {v} | https://openreview.net/forum?id={n.get('forum')}")
