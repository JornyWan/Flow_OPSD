import sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET, time
q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
sort = sys.argv[3] if len(sys.argv) > 3 else "relevance"
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": n, "sortBy": sort, "sortOrder": "descending"})
data=None
for attempt in range(6):
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        break
    except Exception as e:
        print("retry",e,file=sys.stderr); time.sleep(10)
ns = {"a": "http://www.w3.org/2005/Atom"}
root = ET.fromstring(data)
for e in root.findall("a:entry", ns):
    t = " ".join(e.find("a:title", ns).text.split())
    d = e.find("a:published", ns).text[:10]
    i = e.find("a:id", ns).text.split("/abs/")[-1]
    s = " ".join(e.find("a:summary", ns).text.split())
    print(f"[{d}] {i} :: {t}\n    {s[:400]}\n")
