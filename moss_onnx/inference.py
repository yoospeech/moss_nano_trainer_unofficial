"""
ONNX Runtime backend for MOSS Audio Tokenizer.

Drop-in replacement for the PyTorch ``MossAudioTokenizerModel`` that runs
encode / decode through ONNX Runtime — no PyTorch dependency required.

Supports CUDA, TensorRT, and CPU execution providers via ORT's provider list.

Model I/O for the exported 16 kHz Nano checkpoint:
  Encoder: input_values (1,2,T) float32, n_quantizers () int64
           → audio_codes (16,1,T') int64, audio_codes_lengths (1,) int64
  Decoder: audio_codes (16,1,T') int64, n_quantizers () int64
           → audio (1,2,T) float32, audio_lengths (1,) int64
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DOWNSAMPLE_RATE = 1280
SAMPLE_RATE = 16000
N_QUANTIZERS = 16
N_CHANNELS = 2


def _load_ort_session(onnx_path: str | Path, use_gpu: bool = True):
    import onnxruntime as ort

    providers: list[str] = []
    if use_gpu:
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)
    log.info("Loaded %s — providers=%s", Path(onnx_path).name, session.get_providers())
    return session


class OnnxAudioTokenizer:
    """Encode waveforms → audio codes and decode codes → waveforms via ONNX Runtime.

    Args:
        encoder_path: path to encoder ONNX model
        decoder_path: path to decoder ONNX model
        n_quantizers: number of RVQ codebooks (default 16)
        use_gpu: prefer CUDA / TensorRT execution providers
    """

    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        n_quantizers: int = N_QUANTIZERS,
        use_gpu: bool = True,
    ):
        self.n_quantizers = n_quantizers
        self.sample_rate = SAMPLE_RATE
        self.number_channels = N_CHANNELS
        if not 1 <= n_quantizers <= N_QUANTIZERS:
            raise ValueError(f"n_quantizers must be between 1 and {N_QUANTIZERS}")

        encoder_path = Path(encoder_path)
        decoder_path = Path(decoder_path)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {decoder_path}")

        self._encoder = _load_ort_session(encoder_path, use_gpu)
        self._decoder = _load_ort_session(decoder_path, use_gpu)

        self._enc_in = [i.name for i in self._encoder.get_inputs()]
        self._enc_out = [o.name for o in self._encoder.get_outputs()]
        self._dec_in = [i.name for i in self._decoder.get_inputs()]
        self._dec_out = [o.name for o in self._decoder.get_outputs()]

        log.info(
            "OnnxAudioTokenizer ready: n_quantizers=%d, sr=%d, downsample=%d",
            n_quantizers, SAMPLE_RATE, DOWNSAMPLE_RATE,
        )

    def encode(self, waveform: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Encode a waveform to audio codes.

        Args:
            waveform: float32 array, shape ``(T,)``, ``(C, T)``, or ``(1, C, T)``.
                      Mono input is duplicated to stereo.
            n_quantizers: RVQ layers to use (default: ``self.n_quantizers``)

        Returns:
            audio_codes: int64 array, shape ``(T', n_vq)``
        """
        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if not 1 <= n_quantizers <= N_QUANTIZERS:
            raise ValueError(f"n_quantizers must be between 1 and {N_QUANTIZERS}")

        if waveform.ndim == 1:
            waveform = np.repeat(waveform[np.newaxis, :], N_CHANNELS, axis=0)
            waveform = waveform[np.newaxis, :]
        elif waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = np.repeat(waveform, N_CHANNELS, axis=0)
            elif waveform.shape[0] != N_CHANNELS:
                raise ValueError(f"Expected 1 or {N_CHANNELS} channels, got {waveform.shape[0]}")
            waveform = waveform[np.newaxis, :]
        elif waveform.ndim == 3:
            if waveform.shape[0] != 1 or waveform.shape[1] != N_CHANNELS:
                raise ValueError(f"Expected waveform shape (1, {N_CHANNELS}, T), got {waveform.shape}")
        else:
            raise ValueError(f"Expected 1D, 2D, or 3D waveform, got {waveform.ndim}D")

        T = waveform.shape[-1]
        padded = ((T + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != T:
            waveform = np.concatenate(
                [waveform, np.zeros((1, N_CHANNELS, padded - T), dtype=np.float32)], axis=-1,
            )

        waveform = waveform.astype(np.float32)
        nq = np.array(n_quantizers, dtype=np.int64)

        outputs = self._encoder.run(
            self._enc_out,
            {self._enc_in[0]: waveform, self._enc_in[1]: nq},
        )
        audio_codes = outputs[0]       # (16, 1, T')
        code_lengths = outputs[1]      # (1,)

        code_len = int(code_lengths[0])
        codes = audio_codes[:, 0, :code_len]  # (16, T')
        return codes.T.astype(np.int64)        # (T', 16)

    def decode(self, audio_codes: np.ndarray, n_quantizers: int | None = None) -> np.ndarray:
        """Decode audio codes to a waveform.

        Args:
            audio_codes: int64 array, shape ``(T', n_vq)`` or ``(n_vq, T')``
            n_quantizers: RVQ layers (default: ``self.n_quantizers``)

        Returns:
            waveform: float32 array, shape ``(2, T)``
        """
        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if audio_codes.ndim == 2:
            if audio_codes.shape[1] == self.n_quantizers and audio_codes.shape[0] != self.n_quantizers:
                audio_codes = audio_codes.T
            codes_3d = audio_codes[:, np.newaxis, :]
        elif audio_codes.ndim == 3:
            codes_3d = audio_codes
        else:
            raise ValueError(f"Expected 2D or 3D codes, got {audio_codes.ndim}D")

        codes_3d = codes_3d.astype(np.int64)
        nq = np.array(n_quantizers, dtype=np.int64)

        outputs = self._decoder.run(
            self._dec_out,
            {self._dec_in[0]: codes_3d, self._dec_in[1]: nq},
        )
        audio = outputs[0]          # (1, 2, T)
        audio_lengths = outputs[1]  # (1,)

        length = int(audio_lengths[0])
        return audio[0, :, :length].astype(np.float32)


def main() -> None:
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Reconstruct a WAV with the ONNX MOSS Nano codec")
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("hf_export/latest/onnx"))
    parser.add_argument("--n-quantizers", type=int, default=N_QUANTIZERS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    waveform, sample_rate = sf.read(args.input_wav, dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz input, got {sample_rate} Hz")
    waveform = waveform.T
    original_length = waveform.shape[-1]

    tokenizer = OnnxAudioTokenizer(
        args.model_dir / "encoder.onnx",
        args.model_dir / "decoder.onnx",
        n_quantizers=args.n_quantizers,
        use_gpu=not args.cpu,
    )
    codes = tokenizer.encode(waveform)
    reconstructed = tokenizer.decode(codes)[:, :original_length]

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, reconstructed.T, SAMPLE_RATE)
    print(f"codes: {codes.shape}")
    print(f"audio: {reconstructed.shape} @ {SAMPLE_RATE} Hz")
    print(f"saved: {args.output_wav}")


if __name__ == "__main__":
    main()
