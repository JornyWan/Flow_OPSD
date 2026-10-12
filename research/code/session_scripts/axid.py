import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({'id_list': sys.argv[1], 'max_results': 50})
data = urllib.request.urlopen(url, timeout=60).read()
ns = {'a':'http://www.w3.org/2005/Atom'}
for e in ET.fromstring(data).findall('a:entry', ns):
    t = ' '.join(e.find('a:title', ns).text.split()); d = e.find('a:published', ns).text[:10]
    i = e.find('a:id', ns).text.split('/abs/')[-1]; s = ' '.join(e.find('a:summary', ns).text.split())
    print(f"[{d}] {i} | {t}\n    {s}\n")
