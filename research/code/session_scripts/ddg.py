import sys,urllib.request,urllib.parse,re,html,time,random
q=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 12
t=''
for a in range(5):
    try:
        req=urllib.request.Request("https://html.duckduckgo.com/html/",data=urllib.parse.urlencode({'q':q}).encode(),headers={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'})
        t=urllib.request.urlopen(req,timeout=60).read().decode('utf8','ignore')
        if 'result__a' in t: break
    except Exception as e: pass
    time.sleep(5+random.random()*10)
res=re.findall(r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>.*?<a class="result__snippet"[^>]*>(.*?)</a>',t,re.S)
for link,title,sn in res[:n]:
    link=urllib.parse.unquote(re.sub(r'.*uddg=','',link).split('&rut=')[0])
    print(f"- {html.unescape(re.sub('<.*?>','',title))}\n  {link}\n  {html.unescape(re.sub('<.*?>','',sn))[:300]}\n")
if not res: print("NO RESULTS", len(t))
