import csv, re, collections
rows=list(csv.DictReader(open('score_position_deep.csv')))
# dedupe by (prompt_id, seed)
d={}
for r in rows: d[(r['prompt_id'],int(r['seed']))]=r
rel_re=re.compile(r'a photo of an? (.+?) (left of|right of|above|below) an? (.+)$')
by_rel=collections.defaultdict(lambda:[0,0]); sat=collections.defaultdict(collections.Counter)
per_prompt=collections.defaultdict(list)
for (pid,s),r in d.items():
    m=rel_re.match(r['prompt']); rel=m.group(2)
    per_prompt[(pid,r['prompt'],rel)].append(int(r['correct']))
    by_rel[rel][0]+=int(r['correct']); by_rel[rel][1]+=1
    if r['correct']=='0': sat[rel][r['satisfied']+'/'+r['total']]+=1
print("seeds 0-7 only, per relation:")
b=collections.defaultdict(lambda:[0,0,0])
for (pid,p,rel),v in per_prompt.items():
    pass
for (pid,s),r in d.items():
    if s<8:
        rel=rel_re.match(r['prompt']).group(2); b[rel][0]+=int(r['correct']); b[rel][1]+=1
for k,v in b.items(): print(k, v[0], v[1], round(v[0]/v[1],3))
print("failure satisfied breakdown (all seeds):", {k:dict(v) for k,v in sat.items()})
hard=[(k,sum(v),len(v)) for k,v in per_prompt.items() if len(v)>=32 and sum(v)==0]
print("n hard 0/32:",len(hard)); c=collections.Counter(k[2] for k,_,_ in hard); print(c)
for k,s_,n in hard: print(k[1])
nprompt=collections.Counter(k[2] for k in per_prompt); print("prompts per rel",nprompt)
