import sys, urllib.parse, urllib.request, re, html, time
q = sys.argv[1]; n = sys.argv[2] if len(sys.argv)>2 else "25"
order = sys.argv[3] if len(sys.argv)>3 else ""   # "" = relevance, "-announced_date_first"
url = "https://arxiv.org/search/?" + urllib.parse.urlencode({"query": q, "searchtype": "all", "abstracts": "show", "order": order, "size": n})
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
t = None
for a in range(5):
    try:
        t = urllib.request.urlopen(req, timeout=60).read().decode(); break
    except Exception as e:
        print("retry", e, file=sys.stderr); time.sleep(45)
items = t.split('<li class="arxiv-result">')[1:]
def clean(s): return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())
for it in items:
    aid = re.search(r'arxiv.org/abs/([^"]+)"', it).group(1)
    title = clean(re.search(r'<p class="title is-5 mathjax">(.*?)</p>', it, re.S).group(1))
    m = re.search(r'<span class="abstract-full[^>]*>(.*?)<a class="is-size-7"', it, re.S)
    ab = clean(m.group(1)).replace("△ Less","") if m else ""
    d = re.search(r'Submitted</span>\s*(.*?);', it, re.S)
    print(f"[{clean(d.group(1)) if d else ''}] {aid} :: {title}\n    {ab[:450]}\n")
