#!/bin/bash
# usage: arx.sh "query string" [max]
q="$1"; m="${2:-40}"
enc=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$q")
curl -s -m 60 "https://export.arxiv.org/api/query?search_query=${enc}&sortBy=submittedDate&sortOrder=descending&max_results=${m}" | python3 -c "
import sys,re
t=sys.stdin.read()
for e in re.findall(r'<entry>(.*?)</entry>',t,re.S):
    title=re.sub(r'\s+',' ',re.search(r'<title>(.*?)</title>',e,re.S).group(1))
    d=re.search(r'<published>(.*?)</published>',e).group(1)[:10]
    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]
    print(d,i,title)
"
