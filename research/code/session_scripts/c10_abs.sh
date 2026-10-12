for id in "$@"; do curl -sS -m 30 "https://huggingface.co/api/papers/$id" | python3 -c "
import json,sys
try:
    p=json.load(sys.stdin)
except Exception: print('NA'); sys.exit()
print(p.get('id'),(p.get('publishedAt') or '')[:10],'|',' '.join((p.get('title') or '').split())); print('   ',' '.join((p.get('summary') or '').split())[:1400]); print()"; done
