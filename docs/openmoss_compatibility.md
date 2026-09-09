# OpenMOSS API compatibility

The exported checkpoint uses the same public Transformers API as
`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`. The only model-level difference in
these examples is the 16 kHz sample rate. Always derive lengths from
`model.sampling_rate` instead of hard-coding the official model's 48 kHz rate.

The examples below were run against the locally exported checkpoint.

## Streaming encode/decode and variable-length batches

```python
import torch
from transformers import AutoModel

repo_id = "youspeech/moss_nano_trainer_unofficial"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
audio = torch.randn(2, model.sampling_rate * 6)

# 0.08 seconds at 16 kHz is 1,280 samples, one codec frame.
enc = model.encode(audio.unsqueeze(0), return_dict=True, chunk_duration=0.08)
dec = model.decode(enc.audio_codes, return_dict=True, chunk_duration=0.08)

batch_enc = model.batch_encode(
    [audio, audio[:, : model.sampling_rate * 3]], chunk_duration=0.08
)
codes_list = [
    batch_enc.audio_codes[:, i, : batch_enc.audio_codes_lengths[i]]
    for i in range(batch_enc.audio_codes.shape[1])
]
batch_dec = model.batch_decode(codes_list, chunk_duration=0.08)
```

## Continuous batch streaming decode

```python
import torch
from transformers import AutoModel

repo_id = "youspeech/moss_nano_trainer_unofficial"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
num_quantizers = model.config.quantizer_kwargs["num_quantizers"]
codebook_size = model.config.quantizer_kwargs["codebook_size"]

def codes(frames):
    return torch.randint(0, codebook_size, (num_quantizers, frames))

codes_a0, codes_b0 = codes(2), codes(3)
codes_a1, codes_b1, codes_c0 = codes(2), codes(2), codes(1)
codes_a2, codes_b2, codes_c1 = codes(1), codes(2), codes(2)
codes_b3, codes_c2 = codes(1), codes(1)

out_ab0 = model.batch_decode(
    [codes_a0, codes_b0],
    streaming=True,
    max_batch_size=3,
    reset_stream=True,
)
out_abc1 = model.batch_decode(
    [codes_a1, codes_b1, codes_c0], streaming=True
)
out_abc2 = model.batch_decode(
    [codes_a2, codes_b2, codes_c1],
    streaming=True,
    finalize_indices=[0],
)
out_bc3 = model.batch_decode(
    [codes_b3, codes_c2], streaming=True
)
```

## Verified output shapes

For the first example, the exported model produced:

```text
audio_codes:          (16, 1, 75)
decoded audio:        (1, 2, 96000)
batch audio_codes:    (16, 2, 75)
batch decoded audio:  (2, 2, 96000)
```

The four continuous decode calls produced audio tensors shaped `(2, 2, 3840)`,
`(3, 2, 2560)`, `(3, 2, 2560)`, and `(2, 2, 1280)` respectively.
