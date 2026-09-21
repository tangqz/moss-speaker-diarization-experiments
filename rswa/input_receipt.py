"""Explicit audio/time-marker positions, computed from the effective processor."""
import hashlib,json

def receipt(processor,ids,lengths,mapping):
    if hasattr(lengths,'tolist'):lengths=lengths.tolist()
    if hasattr(mapping,'tolist'):mapping=mapping.tolist()
    n=sum(lengths);span=processor._audio_span_ids(n)
    start=ids.index(processor.audio_token_id)
    assert ids[start:start+len(span)]==span,'Expanded audio/time-marker tokens differ'
    markers=[];i=0
    while i<len(span):
        if span[i]==processor.audio_token_id:i+=1;continue
        j=i+1
        while j<len(span) and span[j]!=processor.audio_token_id:j+=1
        seconds=processor.tokenizer.decode(span[i:j],skip_special_tokens=False)
        markers.append(dict(seconds=int(seconds),token_start=start+i,token_end=start+j,ids=span[i:j]))
        i=j
    assert processor.audio_tokens_per_second==12.5
    assert processor.time_marker_every_seconds==5 and processor.enable_time_marker
    return dict(prompt_token_ids=ids,prompt_length=len(ids),audio_tokens=n,
                audio_span_start=start,audio_span_end=start+len(span),markers=markers,marker_count=len(markers),
                audio_feature_lengths=lengths,audio_chunk_mapping=mapping,
                effective_processor=dict(audio_tokens_per_second=processor.audio_tokens_per_second,
                  time_marker_every_seconds=processor.time_marker_every_seconds,
                  enable_time_marker=processor.enable_time_marker))
