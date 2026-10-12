import sys, re, html, urllib.request, urllib.parse, time, base64
for q in sys.argv[1:]:
    url='https://www.bing.com/search?'+urllib.parse.urlencode({'q':q,'count':'20'})
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36','Accept-Language':'en-US,en;q=0.9'})
    d=urllib.request.urlopen(req,timeout=60).read().decode('utf8','ignore')
    print('=== BING:',q)
    for blk in d.split('<li class="b_algo"')[1:]:
        h=re.search(r'<h2[^>]*>(.*?)</h2>',blk,re.S)
        t=html.unescape(re.sub('<.*?>','',h.group(1))) if h else ''
        u=re.search(r'href="([^"]+)"',h.group(1)) if h else None
        u=html.unescape(u.group(1)) if u else ''
        m=re.search(r'[?&]u=a1([^&]+)',u)
        if m:
            s=m.group(1); s+= '='*(-len(s)%4)
            try: u=base64.urlsafe_b64decode(s).decode()
            except: pass
        p=re.search(r'<p[^>]*>(.*?)</p>',blk,re.S)
        p=' '.join(html.unescape(re.sub('<.*?>','',p.group(1))).split())[:220] if p else ''
        print(f'- {t} | {u}\n    {p}')
    print(flush=True); time.sleep(3)
