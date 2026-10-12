import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time
q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
sort = sys.argv[3] if len(sys.argv) > 3 else "relevance"
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": n, "sortBy": sort, "sortOrder": "descending"})
data=None
for attempt in range(4):
    try:
        data = urllib.request.urlopen(url, timeout=60).read(); break
    except Exception as e:
        print("ERR", e, file=sys.stderr); time.sleep(6)
if data is None: sys.exit(1)
ns = {"a": "http://www.w3.org/2005/Atom"}
root = ET.fromstring(data)
print("### Q:", q)
for e in root.findall("a:entry", ns):
    t = " ".join(e.find("a:title", ns).text.split())
    i = e.find("a:id", ns).text.split("/abs/")[-1]
    d = e.find("a:published", ns).text[:10]
    s = " ".join(e.find("a:summary", ns).text.split())
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 350
    print(f"[{d}] {i} | {t}\n    {s[:L]}\n")
