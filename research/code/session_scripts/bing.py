import sys,urllib.request,urllib.parse,re,html,time,random
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 12
url="https://www.bing.com/search?"+urllib.parse.urlencode({'q':q,'count':30})
for a in range(5):
    try:
        t=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36','Accept-Language':'en-US,en;q=0.9'}),timeout=60).read().decode('utf8','ignore'); break
    except Exception as e: time.sleep(5+random.random()*10)
blocks=re.findall(r'<li class="b_algo"(.*?)</li>',t,re.S)
k=0
for b in blocks:
    m=re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',b,re.S)
    if not m: continue
    link=html.unescape(m.group(1)); title=html.unescape(re.sub('<.*?>','',m.group(2)))
    sn=re.search(r'<p[^>]*>(.*?)</p>',b,re.S); sn=html.unescape(re.sub('<.*?>','',sn.group(1))) if sn else ''
    print(f"- {title}\n  {link}\n  {sn[:300]}\n"); k+=1
    if k>=n: break
if k==0: print("NO RESULTS; len",len(t))
