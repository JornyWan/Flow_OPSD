import sys, json, urllib.request
for aid in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"https://api.datacite.org/dois/10.48550/arxiv.{aid}", timeout=90))
        at = d["data"]["attributes"]
        ab = [x["description"] for x in at.get("descriptions") or [] if x.get("descriptionType")=="Abstract"]
        print(f"== {aid} [{at['created'][:10]}] {at['titles'][0]['title']}\n{' '.join((ab[0] if ab else '').split())}\n")
    except Exception as e: print(aid, "ERR", e)
