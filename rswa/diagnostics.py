"""Output diagnostics and token types; these never modify training or parsing."""
import re
import unicodedata
from collections import Counter

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
    spans=Counter((m[1],m[2],m[3],m[4]) for m in legal)
    repeated=max(spans.values(),default=0)
    cursor=0;unparsed=[]
    for match in legal:
        unparsed.append(raw[cursor:match.start()]);cursor=match.end()
    unparsed.append(raw[cursor:])
    unparsed_chars=len(normalize(TAG.sub('', ''.join(unparsed))))
    period_best=dict(period=0,start=0,length=0,cycles=0)
    ids=pred['generated_ids']
    for period in range(1,65):
        matched=0
        for i in range(period,len(ids)):
            matched=matched+1 if ids[i]==ids[i-period] else 0
            length=matched+period
            if matched and length>period_best['length'] and length>=max(128,8*period):
                period_best=dict(period=period,start=i-length+1,length=length,cycles=length/period)
    return dict(official_characters=official,legal_scan_characters=recovered,
        major_parse_loss=(recovered>2*max(official,1) and recovered>100) or unparsed_chars>max(500,.2*recovered),
        longest_identical_run=dict(start=start,count=count,token=token),
        longest_short_period=period_best,repeated_identical_timestamped_segments=repeated,
        timestamp_loop=repeated>=8,unparsed_characters=unparsed_chars,
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
