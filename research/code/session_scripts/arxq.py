import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time
q = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
sort = sys.argv[3] if len(sys.argv) > 3 else "relevance"
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q, "max_results": n, "sortBy": sort})
for attempt in range(6):
    try:
        data = urllib.request.urlopen(url, timeout=60).read(); break
    except Exception as e:
        print('ERR', repr(e)); time.sleep(20)
else:
    sys.exit(1)
ns = {"a": "http://www.w3.org/2005/Atom"}
root = ET.fromstring(data)
for e in root.findall("a:entry", ns):
    t = " ".join(e.find("a:title", ns).text.split())
    d = e.find("a:published", ns).text[:10]
    i = e.find("a:id", ns).text.split("/abs/")[-1]
    s = " ".join(e.find("a:summary", ns).text.split())
    print(f"[{d}] {i} | {t}\n    {s[:450]}\n")
