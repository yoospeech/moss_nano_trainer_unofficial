# Local MOSS Audio Tokenizer Nano model

`modeling_moss_audio_tokenizer.py`, `configuration_moss_audio_tokenizer.py`,
and `config.json` were copied from the existing sibling workspace
`moss_nano_train_unofficial3/third_party/MOSS-Audio-Tokenizer-Nano`.
They retain that local implementation, including its Transformers compatibility
helpers. This is not a newly downloaded or revision-verified upstream snapshot.

Original source: https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano

The source headers and Apache-2.0 license are retained. No pretrained weights
are included. The JSON here remains the original 48 kHz template;
`../moss_trainable.py` constructs the 16 kHz variant from the YAML training
configuration without changing the original model files.
