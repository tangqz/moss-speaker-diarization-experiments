from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained(
    "/work/qt28/moss/models/MOSS-Transcribe-Diarize",
    trust_remote_code=True,
)
print(t.convert_ids_to_tokens(47815))
print(repr(t.decode([47815])))
print(t.special_tokens_map)
