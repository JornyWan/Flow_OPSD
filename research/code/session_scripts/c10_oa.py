import json,sys,urllib.request,urllib.parse
q=sys.argv[1]; frm=sys.argv[2] if len(sys.argv)>2 else "2024-01-01"
u="https://api.openalex.org/works?"+urllib.parse.urlencode({"filter":f"title_and_abstract.search:{q},from_publication_date:{frm}","per_page":25,"sort":"relevance_score:desc"})
d=json.load(urllib.request.urlopen(u,timeout=60))
print("### OA:",q, d.get('meta',{}).get('count'))
for w in d.get('results',[]):
    print(w.get('publication_date'),'|',w.get('display_name'),'|',(w.get('primary_location') or {}).get('landing_page_url'))
