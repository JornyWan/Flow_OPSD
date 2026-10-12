import sys, json, subprocess
for fid in sys.argv[1:]:
    r = subprocess.run(['curl','-sS','-m','40',f'https://api2.openreview.net/notes?id={fid}'],capture_output=True).stdout
    d = json.loads(r)
    for x in d.get('notes',[]):
        c=x['content']; g=lambda k: c.get(k,{}).get('value','') if isinstance(c.get(k),dict) else c.get(k,'')
        print('=== ', g('title'), '|', g('venue'), '|', fid)
        print(' '.join(str(g('abstract')).split())); print('pdf:', g('pdf'), '| arxiv?', g('_bibtex')[:300] if g('_bibtex') else ''); print()
