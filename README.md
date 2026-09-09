<div align="center">
  <h1>Unofficial MOSS Audio Tokenizer Nano Trainer</h1>
  <p>Train and export a 16 kHz MOSS Nano neural audio codec.</p>

  <a href="https://github.com/yoospeech/moss_nano_trainer_unofficial"><img src="https://img.shields.io/badge/GitHub-Trainer-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/youspeech/moss_nano_trainer_unofficial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face model"></a>
  <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"><img src="https://img.shields.io/badge/OpenMOSS-Upstream-4B8BBE" alt="OpenMOSS upstream"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/API%20compatibility-tested-success" alt="API compatibility tested">
</div>

> [!IMPORTANT]
> This is a **16 kHz model trained from random initialization**. The official
> OpenMOSS checkpoint is a 48 kHz stereo model; its boundary-layer weights are
> not shape-compatible with this trainer's 16 kHz boundary layers.

This repository provides RVQ training, mel reconstruction pretraining,
adversarial fine-tuning, Hugging Face export, and checkpoint inference. It is
not affiliated with or endorsed by the OpenMOSS team.

## At a glance

| Property | This model | Official MOSS Nano |
| --- | --- | --- |
| Sample rate | **16 kHz** | 48 kHz |
| Channels | 2 by default; mono supported | 2 |
| Frame rate | 12.5 Hz | 12.5 Hz |
| RVQ | 16 × 1,024 entries | 16 × 1,024 entries |
| Maximum bitrate | 2 kbps | 2 kbps |
| Pretrained initialization | No | Yes |
| Public API | `AutoModel`, `encode`, `decode`, streaming | Same |

## Quick start

```bash
python3 -m pip install -r requirements.txt
hf download youspeech/moss_nano_trainer_unofficial moss_nano_16khz.ckpt \
  --local-dir checkpoints

CKPT=checkpoints/moss_nano_16khz.ckpt \
INPUT_DIR=samples/input \
OUTPUT_DIR=recon_wavs_channel1 \
./inference.sh
```

The exported Transformers model can also be loaded directly:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "youspeech/moss_nano_trainer_unofficial",
    trust_remote_code=True,
).eval()
```

## Features

- 16 kHz mono or stereo training at 12.5 codec frames per second
- 16 residual codebooks with 1,024 entries each
- L2-normalized latent and code vectors with random RVQ prefix dropout
- 250,000 reconstruction batches followed by 250,000 GAN batches
- resumable Lightning checkpoints and TensorBoard logging
- tested compatibility with the official OpenMOSS public and streaming APIs

## Installation

Install PyTorch and torchaudio for your CUDA runtime, then run:

```bash
python3 -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Development was validated with Python
3.12, PyTorch 2.9.1, and Transformers 4.57.1.

## Dataset

Text manifests contain one `id|/absolute/path/audio.wav` entry per line. JSON
and JSONL manifests use `audio_filepath` and may include `duration`. Audio is
resampled to 16 kHz and cropped or padded to 160,000 samples (10 seconds).
JSON entries shorter than five seconds are filtered by default.

```bash
python3 make_json.py wav_paths.txt train.json
```

## Training

```bash
TRAIN_FILELIST=/path/to/train.json bash train.sh
```

The published development checkpoint was trained with this exact command:

```bash
TRAIN_FILELIST=libritts_train_combined_manifest.json \
LOG_DIR=pl_log/moss_16khz_l2 \
bash train.sh resume_ckpt=null model.moss_nano.l2_normalize=true
```

The manifest is not published because it contains local dataset paths.
`l2_normalize=true` and `resume_ckpt=null` are already the defaults, but they
are shown explicitly to document the run exactly.

Resume with the original model and stage configuration:

```bash
TRAIN_FILELIST=/path/to/train.json \
RESUME_CKPT=/path/to/last.ckpt \
bash train.sh
```

Supported environment variables include `PYTHON`, `TRAIN_FILELIST`,
`VAL_FILELIST`, `TEST_FILELIST`, `BATCH_SIZE`, `DEVICES`, `ACCELERATOR`,
`PRECISION`, `MAX_STEPS`, `LOG_DIR`, and `RESUME_CKPT`.

The reconstruction stage takes one generator optimizer step per batch. The
GAN stage takes one discriminator and one generator step, so Lightning's
`global_step` advances twice per GAN batch. The default Lightning limit is
750,000 optimizer steps; `train_batches` is the actual batch counter.
Validation is disabled by default.

## Inference

```bash
CKPT=/path/to/last.ckpt \
INPUT_DIR=/path/to/wavs \
OUTPUT_DIR=./recon_wavs \
bash inference.sh
```

The development inference command was:

```bash
CKPT=pl_log/moss_16khz_l2/last-v1.ckpt \
INPUT_DIR=test_input \
OUTPUT_DIR=recon_wavs_channel1 \
PYTHON=/home/ysw/Documents/anaconda3/envs/vocos_3.13/bin/python \
./inference.sh
```

For a fresh clone, download the checkpoint and use the included sample input:

```bash
CKPT=checkpoints/moss_nano_16khz.ckpt \
INPUT_DIR=samples/input \
OUTPUT_DIR=recon_wavs_channel1 \
./inference.sh
```

Append `num_quantizers=8` to decode with the first eight codebooks. The
current two-channel development checkpoint can emit opposite-polarity
channels, so inference exports channel index 1 as mono by default. Set
`OUTPUT_CHANNEL=0` to select the other channel. Channel selection only affects
WAV export; the model and checkpoint remain two-channel.

## OpenMOSS API compatibility

The Hugging Face export supports the same public waveform API used by the
official MOSS Audio Tokenizer Nano example: `AutoModel`, `encode`, `decode`,
and RVQ prefix slicing. Replace only the repository ID. This checkpoint reports
16 kHz through `model.sampling_rate`, while the official model reports 48 kHz.

```python
import torchaudio
from transformers import AutoModel

repo_id = "youspeech/moss_nano_trainer_unofficial"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()

wav, sr = torchaudio.load(
    "samples/input/2078_142845_000085_000003_original_24khz.wav"
)
if sr != model.sampling_rate:
    wav = torchaudio.functional.resample(wav, sr, model.sampling_rate)
if wav.shape[0] == 1:
    wav = wav.repeat(model.config.number_channels, 1)
else:
    wav = wav[: model.config.number_channels]

enc = model.encode(wav.unsqueeze(0), return_dict=True)
dec = model.decode(enc.audio_codes, return_dict=True)
torchaudio.save("demo_rec.wav", dec.audio.squeeze(0), model.sampling_rate)

dec_rvq8 = model.decode(enc.audio_codes[:8], return_dict=True)
torchaudio.save("demo_rec_rvq8.wav", dec_rvq8.audio.squeeze(0), model.sampling_rate)
```

The non-streaming API, chunked streaming encode/decode, variable-length batch
encode/decode, and continuous batch streaming decode have all been tested with
the exported 16 kHz checkpoint. See
[OpenMOSS API compatibility](docs/openmoss_compatibility.md) for runnable
versions of the official examples.

## Checkpoint download

The development checkpoint is published at:

- [youspeech/moss_nano_trainer_unofficial](https://huggingface.co/youspeech/moss_nano_trainer_unofficial)

Download it with:

```bash
hf download youspeech/moss_nano_trainer_unofficial moss_nano_16khz.ckpt \
  --local-dir checkpoints
```

## Audio samples

`samples/input/` contains the development utterance at 24 kHz and its 16 kHz
resample. `samples/reconstructed/` contains mono reconstructions exported
from channel index 1. They are qualitative examples, not an evaluation set.

## Training snapshot

TensorBoard timestamps were inspected on 2026-09-10 (Asia/Seoul):

| Metric | Value |
| --- | ---: |
| First logged batch | 0 at 2026-09-08 08:37:27 |
| Latest logged batch | 261,168 at 2026-09-10 07:27:48 |
| Measured wall time | 46 hours 50 minutes |
| Current stage | Stage 2 — adversarial training |
| GAN batches completed | 11,168 |

This is a point-in-time record, not a claim about final model quality.
Checkpoints and TensorBoard event files are excluded from the Git repository.

## License

See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
[moss_nano/LICENSE](moss_nano/LICENSE). Users are responsible for dataset and
sample-audio licensing compliance.

## Citation

This repository builds on the MOSS-TTS and MOSS Audio Tokenizer projects.
Please cite the original work:

```bibtex
@misc{gong2026mossttstechnicalreport,
  title        = {MOSS-TTS Technical Report},
  author       = {Yitian Gong and Botian Jiang and Yiwei Zhao and Yucheng Yuan and Kuangwei Chen and Yaozhou Jiang and Cheng Chang and Dong Hong and Mingshu Chen and Ruixiao Li and Yiyang Zhang and Yang Gao and Hanfu Chen and Ke Chen and Songlin Wang and Xiaogui Yang and Yuqian Zhang and Kexin Huang and ZhengYuan Lin and Kang Yu and Ziqi Chen and Jin Wang and Zhaoye Fei and Qinyuan Cheng and Shimin Li and Xipeng Qiu},
  year         = {2026},
  eprint       = {2603.18090},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  url          = {https://arxiv.org/abs/2603.18090}
}

@misc{gong2026mossaudiotokenizerscalingaudiotokenizers,
  title        = {MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models},
  author       = {Yitian Gong and Kuangwei Chen and Zhaoye Fei and Xiaogui Yang and Ke Chen and Yang Wang and Kexin Huang and Mingshu Chen and Ruixiao Li and Qingyuan Cheng and Shimin Li and Xipeng Qiu},
  year         = {2026},
  eprint       = {2602.10934},
  archivePrefix = {arXiv},
  primaryClass = {cs.SD},
  url          = {https://arxiv.org/abs/2602.10934}
}
```
