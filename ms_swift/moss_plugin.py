"""Minimal MS-Swift registration for the unmodified official MOSS model.

No custom Trainer, forward, attention, optimizer or loss implementation.
The official MOSS processor/collator supplies audio features and target labels.
"""
import os
import sys
import torch
from swift.model import Model, ModelGroup, ModelMeta, ModelLoader, register_model
from swift.model.model_arch import MultiModelKeys, register_model_arch
from swift.template import Template, TemplateMeta, register_template

sys.path.insert(0, os.environ.get('MOSS_ROOT', '/work/qt28/moss/MOSS-Transcribe-Diarize'))
from finetune import DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages
import soundfile as sf
import soxr

MODEL_TYPE = 'moss_transcribe_diarize'


class MossLoader(ModelLoader):
    def get_processor(self, model_dir, config):
        return MossTranscribeDiarizeProcessor.from_pretrained(model_dir, trust_remote_code=True)

    def get_model(self, model_dir, config, processor, model_kwargs):
        model = super().get_model(model_dir, config, processor, model_kwargs)
        model.tie_weights()
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        return model


class MossTemplate(Template):
    support_padding_free = False
    placeholder_tokens = ['<|audio_pad|>']

    def _encode(self, inputs):
        if self.padding_free or self.packing:
            raise ValueError('Initial MOSS baseline requires packing=false and padding_free=false.')
        if len(inputs.audios) != 1 or inputs.images or inputs.videos:
            raise ValueError('Expected exactly one complete meeting audio per example.')
        messages = inputs.messages
        if len(messages) not in (1, 2) or messages[0]['role'] != 'user':
            raise ValueError('Expected one user instruction and optional assistant target.')
        prompt = messages[0]['content'].replace('<audio>', '').strip()
        audio_path = inputs.audios[0]
        if len(messages) == 2 and messages[1]['role'] == 'assistant' and messages[1]['content'] is not None:
            target = messages[1]['content'].strip()
            batch = DataCollator(self.processor, self.max_length)([
                dict(audio=audio_path, prompt=prompt, target=target)])
            if int(batch['labels'][0, -1]) != self.tokenizer.eos_token_id:
                raise ValueError('MOSS target EOS missing; refusing a truncated training example.')
        else:
            text = self.processor.apply_chat_template(
                build_transcription_messages(audio_path, prompt), tokenize=False, add_generation_prompt=True)
            audio, sr = sf.read(audio_path, dtype='float32', always_2d=True)
            audio = audio.mean(axis=1)
            target_sr = int(self.processor.feature_extractor.sampling_rate)
            if sr != target_sr:
                audio = soxr.resample(audio, sr, target_sr)
            batch = self.processor(text=[text], audio=[audio], max_length=self.max_length, return_tensors='pt')
        encoded = dict(input_ids=batch['input_ids'][0].tolist(), loss_scale=None)
        encoded['labels'] = batch['labels'][0].tolist() if self.is_training and 'labels' in batch else None
        for key in ['input_features', 'audio_feature_lengths', 'audio_chunk_mapping']:
            encoded[key] = batch[key]
        return encoded

    def _data_collator_mm_data(self, batch):
        features, lengths, mappings = [], [], []
        for i, row in enumerate(batch):
            features.append(torch.as_tensor(row['input_features']))
            lengths.append(torch.as_tensor(row['audio_feature_lengths'], dtype=torch.long))
            mapping = torch.as_tensor(row['audio_chunk_mapping'], dtype=torch.long)
            if not bool((mapping == 0).all()):
                raise ValueError('Expected a single meeting before batching.')
            mappings.append(mapping + i)
        return dict(input_features=torch.cat(features), audio_feature_lengths=torch.cat(lengths),
                    audio_chunk_mapping=torch.cat(mappings))

    def _data_collator(self, batch, *, padding_to=None):
        encoded = super()._data_collator(batch, padding_to=padding_to)
        if os.environ.get('MOSS_IMPLICIT_CAUSAL') == '1':
            # One complete, unpadded sequence needs only the model's causal mask.
            # Append synthetic tokens on the RIGHT before SP; causal attention
            # makes them invisible to real tokens. Their labels are ignored.
            # Keep this OFF for future custom KV masks or padded multi-item batches.
            mask = encoded.get('attention_mask')
            if mask is None or mask.ndim != 2 or mask.shape[0] != 1 or not bool(mask.all()):
                raise ValueError('Implicit full causal attention requires one unpadded meeting.')
            length = encoded['input_ids'].shape[1]
            extra = (-length) % self.sequence_parallel_size
            if extra:
                encoded['input_ids'] = torch.nn.functional.pad(
                    encoded['input_ids'], (0, extra), value=self.tokenizer.pad_token_id)
                if 'labels' in encoded:
                    encoded['labels'] = torch.nn.functional.pad(encoded['labels'], (0, extra), value=-100)
            # Monotone positions prevent Transformers from treating native SP's
            # -1 padding sentinel as a new packed sequence (which we never use).
            encoded['position_ids'] = torch.arange(length+extra).unsqueeze(0)
            encoded.pop('attention_mask')
        return encoded


register_model_arch(MultiModelKeys(
    arch_name=MODEL_TYPE, language_model=['model.language_model', 'lm_head'],
    vision_tower='model.whisper_encoder', aligner='model.vq_adaptor'))
register_template(TemplateMeta(
    template_type=MODEL_TYPE, prefix=[], prompt=['{{QUERY}}'], chat_sep=None,
    suffix=[['eos_token_id']], template_cls=MossTemplate))
register_model(ModelMeta(
    model_type=MODEL_TYPE,
    model_groups=[ModelGroup([Model('OpenMOSS-Team/MOSS-Transcribe-Diarize',
                                   'OpenMOSS-Team/MOSS-Transcribe-Diarize')])],
    loader=MossLoader, template=MODEL_TYPE, model_arch=MODEL_TYPE,
    architectures=['MossTranscribeDiarizeForConditionalGeneration'],
    is_multimodal=True, tags=['audio'], requires=['soundfile', 'soxr']))
