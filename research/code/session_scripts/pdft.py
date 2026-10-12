import sys, subprocess, io, re
from pypdf import PdfReader
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
pid=sys.argv[1]
h=subprocess.run(['curl','-sS','-m','40','-A',UA,f'https://www.alphaxiv.org/abs/{pid}'],capture_output=True).stdout.decode('utf-8','ignore')
m=re.search(r'citation_pdf_url" content="([^"]+)"',h)
url=m.group(1) if m else f'https://pdfs.assets.alphaxiv.org/{pid}v1.pdf'
b=subprocess.run(['curl','-sS','-m','90','-A',UA,url],capture_output=True).stdout
try:
    r=PdfReader(io.BytesIO(b)); txt='\n'.join((p.extract_text() or '') for p in r.pages)
except Exception as e:
    print('ERR',url,e,b[:200]); sys.exit(1)
open(f'{pid}.txt','w').write(txt); print(pid, url, len(txt), 'chars')
