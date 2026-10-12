import sys, json, urllib.request, urllib.parse
for doi in sys.argv[1:]:
    url = "https://api.openalex.org/works/https://doi.org/" + doi
    try:
        w = json.load(urllib.request.urlopen(url, timeout=40))
    except Exception as e:
        print(doi, "ERR", e); continue
    ab = w.get("abstract_inverted_index") or {}
    pos = sorted((p,k) for k,v in ab.items() for p in v)
    print(f"### {w['title']} [{w.get('publication_date')}] {doi}")
    print(" ".join(k for p,k in pos)); print()
