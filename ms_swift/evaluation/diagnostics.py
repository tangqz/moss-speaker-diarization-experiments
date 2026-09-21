"""Output diagnostics and token types; these never modify training or parsing."""
import re
import unicodedata

LEGAL = re.compile(r'\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]')
TAG = re.compile(r'\[(?:S\d+|\d+(?:\.\d+)?)\]')
TYPES = ['text','timestamp','speaker','eos','boundary','unknown']


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold()
                   if not c.isspace() and unicodedata.category(c)[0] not in 'PSZC')


def longest_run(ids):
    best=(0,0,None)
    i=0
    while i<len(ids):
        j=i+1
        while j<len(ids) and ids[j]==ids[i]: j+=1
        if j-i>best[1]: best=(i,j-i,ids[i])
        i=j
    return best


def output_diagnostics(pred):
    raw=pred['raw_text']
    official=sum(len(normalize(s['text'])) for s in pred['segments'])
    legal=[m for m in LEGAL.finditer(raw) if float(m[4])>=float(m[1]) and m[3].strip()]
    recovered=sum(len(normalize(m[3])) for m in legal)
    start,count,token=longest_run(pred['generated_ids'])
    invalid=[x for x in re.findall(r'\[([^\[\]]+)\]',raw)
             if not re.fullmatch(r'(?:S\d+|\d+(?:\.\d+)?)',x)]
    return dict(official_characters=official,legal_scan_characters=recovered,
        major_parse_loss=recovered>2*max(official,1) and recovered>100,
        longest_identical_run=dict(start=start,count=count,token=token),
        invalid_tags=invalid[:100],invalid_tag_count=len(invalid))


def target_types(text, offsets, ids, eos_id):
    # Offsets describe characters in the original supervised target. No rollout
    # is retokenized, and statistics at target j are attached to query j-1.
    labels=['text']*len(text)
    for m in TAG.finditer(text):
        label='speaker' if m[0].startswith('[S') else 'timestamp'
        labels[m.start():m.end()]=[label]*(m.end()-m.start())
    result=[]
    for token,(start,end) in zip(ids,offsets):
        if token==eos_id:
            result.append(TYPES.index('eos'))
        elif start>=end or start<0 or end>len(labels):
            result.append(TYPES.index('unknown'))
        else:
            kinds=set(labels[start:end])
            result.append(TYPES.index(next(iter(kinds)) if len(kinds)==1 else 'boundary'))
    return result
