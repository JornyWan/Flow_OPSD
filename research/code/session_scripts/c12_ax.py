import sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, random
ns={'a':'http://www.w3.org/2005/Atom'}
def q(query, n=15, sort='relevance', abslen=350):
    url='https://export.arxiv.org/api/query?'+urllib.parse.urlencode({'search_query':query,'max_results':n,'sortBy':sort,'sortOrder':'descending'})
    data=None
    for attempt in range(10):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'lit-check-c12/1.0'})
            data=urllib.request.urlopen(req,timeout=60).read(); break
        except Exception as e:
            time.sleep(4+attempt*4+random.random()*3)
    if data is None: print('FAIL',query); return
    root=ET.fromstring(data)
    tot=root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text
    print(f'=== {query} [{sort}] total={tot}')
    for e in root.findall('a:entry',ns):
        t=' '.join(e.find('a:title',ns).text.split())
        pid=e.find('a:id',ns).text.split('/abs/')[-1]
        d=e.find('a:published',ns).text[:10]
        s=' '.join(e.find('a:summary',ns).text.split())[:abslen]
        print(f'- [{d}] {pid} | {t}\n    {s}')
    print(flush=True)
if __name__=='__main__':
    sort=sys.argv[1]; n=int(sys.argv[2]); abslen=int(sys.argv[3])
    for qq in sys.argv[4:]:
        q(qq,n,sort,abslen); time.sleep(4)
