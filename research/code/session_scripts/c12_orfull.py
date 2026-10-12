import sys, json, urllib.request, urllib.parse, time
for term in sys.argv[1:]:
    url='https://api2.openreview.net/notes/search?'+urllib.parse.urlencode({'term':term,'limit':3,'type':'terms','content':'all','source':'forum'})
    d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read())
    for no in d.get('notes',[])[:2]:
        c=no.get('content',{}); g=lambda k:(c.get(k) or {}).get('value','') if isinstance(c.get(k),dict) else c.get(k,'')
        print('TITLE:',g('title'),'|',g('venue'),'| forum',no.get('forum')); print('AUTHORS:',g('authors')); print('KW:',g('keywords')); print('ABS:',' '.join(str(g('abstract')).split())); print('TLDR:',g('TLDR') or g('TL;DR')); print('PDF:',g('pdf')); print()
    time.sleep(3)
