# Unofficial MOSS Audio Tokenizer Nano Trainer

An unofficial training and inference pipeline for a **16 kHz MOSS Audio
Tokenizer Nano**. It includes RVQ training, mel reconstruction pretraining,
and adversarial fine-tuning with multi-period and multi-resolution STFT
discriminators. This project is not affiliated with the OpenMOSS team.

## Features

- 16 kHz mono or stereo training at 12.5 codec frames per second
- 16 residual codebooks with 1,024 entries each (2 kbps using all codebooks)
- L2-normalized latent and code vectors with random RVQ prefix dropout
- 250,000 reconstruction batches followed by 250,000 GAN batches
- resumable Lightning checkpoints and TensorBoard logging

The 16 kHz boundary layers are not shape-compatible with the official 48 kHz
pretrained model, so this trainer starts from random initialization.

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

Resume with the original model and stage configuration:

```bash
TRAIN_FILELIST=/path/to/train.json \
RESUME_CKPT=/path/to/last.ckpt \
bash train.sh
```

Hydra overrides can be appended:

```bash
TRAIN_FILELIST=/path/to/train.json \
bash train.sh dataset.train.batch_size=4 model.moss_nano.channels=1
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

Append `num_quantizers=8` to decode with the first eight codebooks. The
current two-channel development checkpoint can emit opposite-polarity
channels, so inference exports channel index 1 as mono by default. Set
`OUTPUT_CHANNEL=0` to select the other channel. Channel selection only affects
WAV export; the model and checkpoint remain two-channel.

### OpenMOSS example compatibility

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

## Pretrained checkpoint

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

- first logged batch: 0 at 2026-09-08 08:37:27
- latest logged batch: 261,168 at 2026-09-10 07:27:48
- measured elapsed wall time: 46 hours 50 minutes
- stage 2: 11,168 adversarial batches completed

This is a point-in-time record, not a claim about final model quality.
Checkpoints and TensorBoard event files are excluded from the Git repository.

## Tests

```bash
python3 -m unittest discover -s tests -v
bash -n train.sh inference.sh
```

The tests cover resampling, mono/stereo batches, partial frames, gradients,
normalized code decoding, stage transition, checkpoint restore, and resume.

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
