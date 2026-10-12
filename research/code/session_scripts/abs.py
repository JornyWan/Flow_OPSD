import sys, urllib.request, xml.etree.ElementTree as ET
ids = ",".join(sys.argv[1:])
data = urllib.request.urlopen("https://export.arxiv.org/api/query?id_list="+ids+"&max_results=50", timeout=60).read()
ns = {"a": "http://www.w3.org/2005/Atom"}
for e in ET.fromstring(data).findall("a:entry", ns):
    print("["+e.find("a:published", ns).text[:10]+"]", e.find("a:id", ns).text.split("/abs/")[-1], "::", " ".join(e.find("a:title", ns).text.split()))
    print("   ", " ".join(e.find("a:summary", ns).text.split()), "\n")
