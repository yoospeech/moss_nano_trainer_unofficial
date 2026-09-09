"""Training adapter for the local Nano architecture; upstream state keys are retained."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from moss_nano.configuration_moss_audio_tokenizer import MossAudioTokenizerConfig
from moss_nano.modeling_moss_audio_tokenizer import MossAudioTokenizerModel, MossAudioTokenizerLFQ


class NormalizedMossLFQ(MossAudioTokenizerLFQ):
    """Unit-length code vectors in both nearest-neighbor and code-only decoding.

    Parameter names/shapes remain identical to the upstream LFQ.
    """

    def embed_code(self, embed_id):
        return F.normalize(super().embed_code(embed_id).float(), dim=-1, eps=1e-8)


class TrainableMossAudioTokenizer(nn.Module):
    def __init__(self, sampling_rate=16000, channels=2, frame_rate=12.5,
                 num_quantizers=16, commitment=0.25, codebook_weight=1.0,
                 quantizer_dropout=1.0, l2_normalize=False):
        super().__init__()
        if sampling_rate != 16000 or channels not in (1, 2):
            raise ValueError('This training adapter supports 16 kHz mono or stereo.')
        if frame_rate <= 0:
            raise ValueError('frame_rate must be positive.')
        if not 1 <= num_quantizers <= 16 or not 0 <= quantizer_dropout <= 1:
            raise ValueError('Invalid quantizer count or dropout probability.')
        with Path(__file__).with_name('moss_nano').joinpath('config.json').open() as f:
            config = json.load(f)
        # Internal patches [boundary, 2, 2, 2, 4] operate on interleaved channels.
        boundary = sampling_rate * channels / (frame_rate * 32)
        if boundary < 1 or not boundary.is_integer():
            raise ValueError('frame_rate must yield an integer boundary patch size.')
        boundary = int(boundary)
        config.update(sampling_rate=sampling_rate, sample_rate=sampling_rate,
                      number_channels=channels, downsample_rate=boundary * 32 // channels)
        config['encoder_kwargs'][0]['patch_size'] = boundary
        config['encoder_kwargs'][1]['input_dimension'] = boundary
        config['decoder_kwargs'][-2]['output_dimension'] = boundary
        config['decoder_kwargs'][-1]['patch_size'] = boundary
        config.pop('reversed_decoder_kwargs', None)
        config['quantizer_kwargs']['num_quantizers'] = num_quantizers
        self.codec = MossAudioTokenizerModel(MossAudioTokenizerConfig(**config))
        # Missing flag means the legacy raw-vector adapter for old checkpoints.
        self.l2_normalize = l2_normalize
        if l2_normalize:
            for index, original in enumerate(self.codec.quantizer.quantizers):
                normalized = NormalizedMossLFQ(
                    input_dim=original.input_dim, codebook_size=original.codebook_size,
                    codebook_dim=original.codebook_dim,
                )
                normalized.load_state_dict(original.state_dict())
                self.codec.quantizer.quantizers[index] = normalized
        self.commitment = commitment
        self.codebook_weight = codebook_weight
        self.quantizer_dropout = quantizer_dropout

    def forward(self, wav, lengths=None, num_quantizers=None):
        if wav.ndim != 3 or wav.shape[1] != self.codec.number_channels:
            raise ValueError('Expected waveform [batch, configured channels, samples].')
        if lengths is None:
            lengths = torch.full((wav.shape[0],), wav.shape[-1], device=wav.device, dtype=torch.long)
        if torch.any(lengths <= 0) or torch.any(lengths > wav.shape[-1]):
            raise ValueError('Audio lengths must be positive and no greater than waveform length.')
        hop = self.codec.downsample_rate
        # Include the last partial frame; upstream patch layers use floor division.
        padded_lengths = torch.div(lengths + hop - 1, hop, rounding_mode='floor') * hop
        hidden, hidden_lengths = self.codec._flatten_channels_for_codec(wav, padded_lengths)
        for layer in self.codec.encoder:
            hidden, hidden_lengths = layer(hidden, hidden_lengths)
        quantizer = self.codec.quantizer
        count = quantizer.num_quantizers if num_quantizers is None else int(num_quantizers)
        if not 1 <= count <= quantizer.num_quantizers:
            raise ValueError('Requested codebook prefix is outside the trained range.')
        if self.training and num_quantizers is None and torch.rand(()) < self.quantizer_dropout:
            count = int(torch.randint(1, count + 1, ()).item())
        losses, codes, utilization = [], [], []
        with torch.autocast(device_type=wav.device.type, enabled=False):
            residual = quantizer.input_proj(hidden.float()).float()
            valid = torch.arange(residual.shape[-1], device=wav.device)[None] < hidden_lengths[:, None]
            mask = valid[:, None]
            quantized = torch.zeros_like(residual)
            for layer in quantizer.quantizers[:count]:
                z_e = layer.in_proj(residual * mask).float()
                if self.l2_normalize:
                    z_e = F.normalize(z_e, dim=1, eps=1e-8)
                z_q, indices = layer.decode_latents(z_e)
                denom = (valid.sum() * z_e.shape[1]).clamp_min(1)
                loss = (self.commitment * (z_e - z_q.detach()).square()
                        + self.codebook_weight * (z_q - z_e.detach()).square())
                losses.append((loss * mask).sum() / denom)
                # embed_code supplies the same vectors for training and decode_codes.
                q = layer.out_proj(z_e + (z_q - z_e).detach()).float() * mask
                quantized = quantized + q
                residual = residual - q.detach()
                codes.append(indices.masked_fill(~valid, 0))
                utilization.append(wav.new_tensor(torch.unique(indices[valid]).numel() / layer.codebook_size))
            hidden = quantizer.output_proj(quantized).float()
        for layer in self.codec.decoder:
            hidden, hidden_lengths = layer(hidden, hidden_lengths)
        audio, _ = self.codec._restore_channels_from_codec(hidden, hidden_lengths)
        audio = audio[..., :wav.shape[-1]]
        sample_mask = torch.arange(wav.shape[-1], device=wav.device)[None, None] < lengths[:, None, None]
        return dict(gt_wav=wav * sample_mask, gen_wav=audio * sample_mask,
                    vq_loss=torch.stack(losses), vq_code=torch.stack(codes),
                    utilization=torch.stack(utilization))

    @torch.inference_mode()
    def inference(self, wav, lengths=None, num_quantizers=None):
        return self(wav, lengths, num_quantizers)['gen_wav']
