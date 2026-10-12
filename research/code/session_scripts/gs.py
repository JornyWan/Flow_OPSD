import sys, urllib.request, urllib.parse, re, html, time
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
def gs(q, ylo=2025):
    url = "https://scholar.google.com/scholar?" + urllib.parse.urlencode({'q': q, 'as_ylo': ylo, 'hl':'en', 'num': 20})
    import subprocess
    h = subprocess.run(['curl','-sS','-m','40','-A',UA,url],capture_output=True).stdout.decode('utf-8','ignore')
    if 'gs_captcha' in h or 'not a robot' in h: print('CAPTCHA'); return
    blocks = re.split(r'<div class="gs_r gs_or gs_scl[^"]*"', h)[1:]
    if not blocks: blocks = re.split(r'(?=<h3 class="gs_rt")', h)[1:]
    print(f'### {q}  ({len(blocks)} hits)')
    for b in blocks:
        t = re.search(r'<h3 class="gs_rt"[^>]*>(.*?)</h3>', b, re.S)
        title = html.unescape(re.sub('<[^>]+>','',t.group(1))).strip() if t else '?'
        link = re.search(r'<h3 class="gs_rt"[^>]*>.*?<a[^>]*href="([^"]+)"', b, re.S)
        meta = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', b, re.S)
        meta = html.unescape(re.sub('<[^>]+>','',meta.group(1))).strip() if meta else ''
        sn = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', b, re.S)
        sn = html.unescape(re.sub('<[^>]+>','',sn.group(1))).strip() if sn else ''
        print(f'- {title} | {link.group(1) if link else ""}\n    {meta[:140]}\n    {sn[:330]}')
    print()
for i,q in enumerate(sys.argv[1:]):
    if i: time.sleep(12)
    gs(q)
