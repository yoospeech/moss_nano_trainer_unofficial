import os
import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
import soundfile as sf
from pathlib import Path
from os.path import exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ocwd = (hydra.utils.get_original_cwd() if hydra.core.hydra_config.HydraConfig.initialized() else os.getcwd())
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        dl = DataLoader(ds, batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=int(self.cfg.dataset.get("num_workers", 8)),
                        pin_memory=bool(self.cfg.dataset.get("pin_memory", True)),
                        collate_fn=ds.collate_fn,
                        persistent_workers=int(self.cfg.dataset.get("num_workers", 8)) > 0)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class FSDataset(Dataset):
    """Audio manifests with fixed-length waveform crops and valid lengths.

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = (hydra.utils.get_original_cwd() if hydra.core.hydra_config.HydraConfig.initialized() else os.getcwd())

        self.sr = cfg.preprocess.audio.sr
        self.min_duration_sec = float(cfg.dataset.get('min_duration_sec', 5.0))

        random_val_ratio = float(cfg.dataset.get('random_val_ratio', 0.0))
        if phase in ('train', 'val') and random_val_ratio > 0.0:
            self.filelist = self._load_random_split_entries()
        else:
            filelist_path = join(self.ocwd, self.phase_cfg.filelist)
            self.filelist = self._load_entries(filelist_path)
        self.min_audio_length = cfg.dataset.min_audio_length

    def _load_random_split_entries(self):
        ratio = float(self.cfg.dataset.get('random_val_ratio', 0.0))
        seed = int(self.cfg.dataset.get('random_val_seed', 1024))
        if not (0.0 < ratio < 1.0):
            raise ValueError(f'dataset.random_val_ratio must be in (0, 1), got {ratio}')

        train_filelist_path = join(self.ocwd, self.cfg.dataset.train.filelist)
        all_entries = self._load_entries(train_filelist_path)
        if len(all_entries) < 2:
            raise ValueError(f'Need at least 2 entries for random split, got {len(all_entries)}')

        indices = list(range(len(all_entries)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        val_count = max(1, int(round(len(all_entries) * ratio)))
        val_indices = set(indices[:val_count])
        train_entries = [all_entries[i] for i in range(len(all_entries)) if i not in val_indices]
        val_entries = [all_entries[i] for i in range(len(all_entries)) if i in val_indices]

        if len(train_entries) == 0:
            raise ValueError('Random split produced empty train set; reduce dataset.random_val_ratio')

        if self.phase == 'train':
            return train_entries
        return val_entries

    def _resolve_audio_path(self, wavpath):
        if os.path.isabs(wavpath):
            return wavpath
        if exists(join(self.ocwd, wavpath)):
            return join(self.ocwd, wavpath)
        return join(self.cfg.preprocess.datasets.LibriSpeech.root, wavpath)

    def _get_duration_sec(self, item, wavpath):
        for key in ('duration', 'duration_sec', 'audio_duration'):
            if isinstance(item, dict) and item.get(key) is not None:
                try:
                    return float(item[key])
                except (TypeError, ValueError):
                    pass

        resolved_path = self._resolve_audio_path(wavpath)
        try:
            info = torchaudio.info(resolved_path)
            if info.sample_rate > 0:
                return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            try:
                info = sf.info(resolved_path)
                return info.frames / info.samplerate
            except (RuntimeError, OSError):
                return None

        return None

    def _entries_from_manifest_objects(self, objects):
        entries = []
        for idx, item in enumerate(objects):
            if not isinstance(item, dict):
                continue
            wavpath = item.get('audio_filepath') or item.get('audio_path') or item.get('wav_path') or item.get('path')
            if wavpath is None:
                continue
            duration_sec = self._get_duration_sec(item, wavpath)
            if duration_sec is None or duration_sec < self.min_duration_sec:
                continue
            fid = item.get('id') or item.get('uid') or item.get('utt_id') or Path(wavpath).stem or str(idx)
            entries.append((str(fid), str(wavpath)))
        return entries

    def _load_entries(self, filelist_path):
        lower = filelist_path.lower()
        if lower.endswith('.json'):
            with open(filelist_path, encoding='utf8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ('items', 'data', 'manifest', 'entries'):
                    if key in data and isinstance(data[key], list):
                        return self._entries_from_manifest_objects(data[key])
                raise ValueError(f'Unsupported manifest dict schema in {filelist_path}')
            if isinstance(data, list):
                entries = self._entries_from_manifest_objects(data)
                if len(entries) == 0:
                    raise ValueError(f'No valid `audio_filepath` entries found in {filelist_path}')
                return entries
            raise ValueError(f'Unsupported manifest schema in {filelist_path}')

        if lower.endswith('.jsonl'):
            items = []
            with open(filelist_path, encoding='utf8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
            entries = self._entries_from_manifest_objects(items)
            if len(entries) == 0:
                raise ValueError(f'No valid `audio_filepath` entries found in {filelist_path}')
            return entries

        return utils.read_filelist(filelist_path)

    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        channels = int(self.cfg.preprocess.audio.get('channels', 1))
        if channels > 1:
            wav, _ = librosa.load(path, sr=self.sr, mono=False)
            if wav.ndim == 1:
                wav = np.repeat(wav[None], channels, axis=0)
            return wav[:channels]
        wav, sr = librosa.load(path, sr=self.sr)
        wav = librosa.effects.trim(wav, top_db=30)[0]
        #wav = wav.clip(-1.0, 1.0)
        return wav

    def __getitem__(self, idx):
        (fid, wavpath) = self.filelist[idx]
        wavpath = self._resolve_audio_path(wavpath)
        wav = self.load_wav(wavpath)
        wav = torch.from_numpy(wav)
        if wav.shape[-1] == 0:
            raise ValueError(f'Empty audio after loading: {wavpath}')
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        valid_length = min(wav.shape[-1], self.min_audio_length)
        length = wav.shape[-1]
        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[-1]
        i = random.randint(0, length-self.min_audio_length) if self.phase == 'train' else (length-self.min_audio_length) // 2
        wav = wav[..., i:i+self.min_audio_length]

        out = {
            'fid': fid,
            'wav': wav,
            'length': valid_length,
        }

        return out

    def collate_fn(self, bs):
        fids = [b['fid'] for b in bs]
        wavs = [b['wav'] for b in bs]
        lengths = [b['length'] for b in bs]
        wavs = torch.stack(wavs)
        lengths = torch.LongTensor(lengths)

        out = {
            'fid': fids,
            'wav': wavs,
            'lengths': lengths,
        }
        return out
