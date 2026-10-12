import sys, re, html, urllib.request, urllib.parse, time, random
def s(query, size=25, order='', abslen=350, field='all'):
    params={'query':query,'searchtype':field,'size':size,'abstracts':'show'}
    if order: params['order']=order
    url='https://arxiv.org/search/?'+urllib.parse.urlencode(params)
    data=None
    for a in range(6):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            data=urllib.request.urlopen(req,timeout=60).read().decode('utf8','ignore'); break
        except Exception as e:
            time.sleep(10+10*a)
    if data is None: print('FAIL',query); return
    m=re.search(r'Showing 1&ndash;\d+ of ([\d,]+)',data)
    print(f'=== {query} [{order or "rel"}] total={m.group(1) if m else 0}')
    for li in data.split('<li class="arxiv-result">')[1:]:
        pid=re.search(r'arxiv.org/abs/([\w.]+)',li); pid=pid.group(1) if pid else '?'
        t=re.search(r'<p class="title is-5 mathjax">(.*?)</p>',li,re.S)
        t=' '.join(html.unescape(re.sub('<.*?>','',t.group(1))).split()) if t else '?'
        ab=re.search(r'<span class="abstract-full[^>]*>(.*?)<a',li,re.S)
        ab=' '.join(html.unescape(re.sub('<.*?>','',ab.group(1))).split())[:abslen] if ab else ''
        d=re.search(r'Submitted</span>\s*([^;]+);',li); d=d.group(1).strip() if d else ''
        print(f'- [{d}] {pid} | {t}\n    {ab}')
    print(flush=True)
if __name__=='__main__':
    order=sys.argv[1]; size=int(sys.argv[2]); abslen=int(sys.argv[3])
    if order=='rel': order=''
    for qq in sys.argv[4:]:
        s(qq,size,order,abslen); time.sleep(8)
