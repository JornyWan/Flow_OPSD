import sys, json, urllib.request, urllib.parse, time
q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv)>2 else 25
frm = sys.argv[3] if len(sys.argv)>3 else "2023-01-01"
params = {"filter": f"title_and_abstract.search:{q},from_publication_date:{frm}", "per-page": n, "sort":"publication_date:desc"}
url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
for attempt in range(3):
    try:
        d = json.load(urllib.request.urlopen(url, timeout=40)); break
    except Exception as e:
        print("ERR", e); time.sleep(5)
else:
    sys.exit()
print("TOTAL", d["meta"]["count"])
seen=set()
for w in d["results"]:
    t = w.get('title')
    if t in seen: continue
    seen.add(t)
    ab = w.get("abstract_inverted_index")
    txt = ""
    if ab:
        pos = []
        for k,v in ab.items():
            for p in v: pos.append((p,k))
        txt = " ".join(k for p,k in sorted(pos))
    print(f"- [{w.get('publication_date')}] {t} | {(w.get('primary_location') or {}).get('landing_page_url','')}")
    if txt and len(sys.argv)>4: print("    ", txt[:int(sys.argv[4])])
