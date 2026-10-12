import sys, json, urllib.request, urllib.parse, time
# usage: term  title_substring
term, sub = sys.argv[1], sys.argv[2].lower()
url='https://api2.openreview.net/notes/search?'+urllib.parse.urlencode({'term':term,'limit':25,'type':'terms','content':'all','source':'forum'})
d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read())
for no in d.get('notes',[]):
    c=no.get('content',{}); g=lambda k:(c.get(k) or {}).get('value','') if isinstance(c.get(k),dict) else c.get(k,'')
    if sub in str(g('title')).lower():
        print('TITLE:',g('title'),'|',g('venue'),'| forum',no.get('forum')); print('AUTHORS:',g('authors')); print('ABS:',' '.join(str(g('abstract')).split())); print('PDF:',g('pdf')); print(); break
