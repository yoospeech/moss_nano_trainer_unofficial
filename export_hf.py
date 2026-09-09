"""Export a Lightning checkpoint as a Hugging Face AutoModel repository."""

import argparse
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from inference import _build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('hyper_parameters', {}).get('cfg')
    if config is None:
        raise KeyError('The Lightning checkpoint does not contain its Hydra configuration.')
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    adapter = _build_model(config, checkpoint, 'cpu')
    # Training normalizes code vectors at every lookup. Materialize that
    # normalization so the stock Hugging Face class decodes identically.
    with torch.no_grad():
        for quantizer in adapter.codec.quantizer.quantizers:
            quantizer.codebook.weight.copy_(
                torch.nn.functional.normalize(quantizer.codebook.weight.float(), dim=-1)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter.codec.save_pretrained(args.output_dir, safe_serialization=True)
    source_dir = Path(__file__).with_name('moss_nano')
    for name in ('configuration_moss_audio_tokenizer.py',
                 'modeling_moss_audio_tokenizer.py', '__init__.py', 'LICENSE'):
        shutil.copy2(source_dir / name, args.output_dir / name)
    print(f'Exported Hugging Face model to {args.output_dir}')


if __name__ == '__main__':
    main()
