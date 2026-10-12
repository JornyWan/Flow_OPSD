import sys, json, urllib.request, urllib.parse, time
q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv)>2 else 25
url = "https://api.datacite.org/dois?" + urllib.parse.urlencode({"query": q, "page[size]": n, "sort": "-created", "client-id":"arxiv.content"})
for attempt in range(3):
    try:
        d = json.load(urllib.request.urlopen(url, timeout=60)); break
    except Exception as e:
        print("ERR", e); time.sleep(5)
else:
    sys.exit()
print("TOTAL", d["meta"]["total"])
for it in d["data"]:
    a = it["attributes"]
    t = a["titles"][0]["title"] if a["titles"] else ""
    desc = ""
    for de in a.get("descriptions") or []:
        if de.get("descriptionType")=="Abstract": desc = de.get("description","")
    print(f"- [{a.get('created','')[:10]}] {t} | {a.get('doi')}")
    if len(sys.argv)>3: print("    ", desc[:int(sys.argv[3])].replace("\n"," "))
