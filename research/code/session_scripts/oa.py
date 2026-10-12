import sys, urllib.parse, urllib.request, json
q = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv)>2 else 15
frm = sys.argv[3] if len(sys.argv)>3 else "2023-01-01"
url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": q, "per-page": n, "filter": f"from_publication_date:{frm}"})
d = json.load(urllib.request.urlopen(url, timeout=60))
def ab(inv):
    if not inv: return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps: pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))
for w in d["results"]:
    print(f"[{w['publication_date']}] {w['title']}  ({(w.get('primary_location') or {}).get('landing_page_url')})")
    print("   ", ab(w.get("abstract_inverted_index"))[:350], "\n")
