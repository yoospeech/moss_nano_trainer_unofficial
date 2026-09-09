import os
from glob import glob
from os.path import basename, join
from time import time

import hydra
import librosa
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from moss_trainable import TrainableMossAudioTokenizer


def _extract_generator_state_dict(checkpoint):
    if 'generator' in checkpoint and isinstance(checkpoint['generator'], dict):
        return checkpoint['generator']
    state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise KeyError('Checkpoint has no generator or Lightning state_dict.')
    prefix = 'model.generator.'
    generator_state = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not generator_state:
        raise KeyError(f'Checkpoint has no parameters with prefix {prefix!r}.')
    return generator_state


def _build_model(cfg, checkpoint, device):
    model = TrainableMossAudioTokenizer(**cfg.model.moss_nano)
    model.load_state_dict(_extract_generator_state_dict(checkpoint), strict=True)
    return model.eval().to(device)


def _load_audio(path, sample_rate, channels):
    wav, _ = librosa.load(path, sr=sample_rate, mono=False)
    if wav.ndim == 1:
        wav = wav[None]
    if channels == 1:
        wav = wav.mean(axis=0, keepdims=True)
    elif wav.shape[0] == 1:
        wav = wav.repeat(channels, axis=0)
    return torch.from_numpy(wav[:channels]).unsqueeze(0)


@hydra.main(config_path='config', config_name='moss_16khz', version_base=None)
def main(cfg):
    if cfg.ckpt is None or cfg.input_dir is None or cfg.output_dir is None:
        raise ValueError('Set ckpt, input_dir, and output_dir.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading checkpoint: {cfg.ckpt}')
    checkpoint = torch.load(cfg.ckpt, map_location='cpu', weights_only=False)
    model_cfg = checkpoint.get('hyper_parameters', {}).get('cfg', cfg)
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    model = _build_model(model_cfg, checkpoint, device)
    sample_rate = model.codec.sampling_rate
    channels = model.codec.number_channels

    os.makedirs(cfg.output_dir, exist_ok=True)
    wav_paths = glob(join(cfg.input_dir, '*.wav'))
    print(f'Found {len(wav_paths)} WAV file(s) in {cfg.input_dir}')

    started = time()
    for wav_path in tqdm(wav_paths):
        wav = _load_audio(wav_path, sample_rate, channels).to(device)
        recon = model.inference(wav, num_quantizers=cfg.get('num_quantizers'))
        output_channel = int(cfg.get('output_channel', 1))
        if not 0 <= output_channel < recon.shape[1]:
            raise ValueError(
                f'output_channel={output_channel} is out of range for '
                f'{recon.shape[1]} generated channel(s).'
            )
        output = recon[0, output_channel].detach().cpu().numpy()
        sf.write(join(cfg.output_dir, basename(wav_path)), output, sample_rate)

    print(f'Inference finished in {(time() - started) / 60:.2f} minutes')


if __name__ == '__main__':
    main()
