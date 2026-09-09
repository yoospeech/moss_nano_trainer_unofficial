"""CPU integration checks: python -m unittest discover -s tests -v."""
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pytorch_lightning as pl
from hydra import compose, initialize

from data_module import DataModule, FSDataset
from lightning_module import CodecLightningModule
from inference import _build_model
from moss_trainable import TrainableMossAudioTokenizer


class MossTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gradients_and_upstream_code_decode(self):
        model = TrainableMossAudioTokenizer(quantizer_dropout=0)
        wav = torch.randn(1, 2, 2560)
        output = model(wav, num_quantizers=2)
        (output['gen_wav'].square().mean() + output['vq_loss'].sum()).backward()
        for component in (model.codec.encoder, model.codec.decoder,
                          model.codec.quantizer.quantizers[0].codebook):
            grads = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        model.eval()
        with torch.no_grad():
            hidden = model.codec.quantizer.decode_codes(output['vq_code'])
            lengths = torch.tensor([hidden.shape[-1]])
            for layer in model.codec.decoder:
                hidden, lengths = layer(hidden, lengths)
            decoded, _ = model.codec._restore_channels_from_codec(hidden, lengths)
        torch.testing.assert_close(output['gen_wav'], decoded, atol=2e-5, rtol=2e-5)
        tail = model(torch.randn(1, 2, 2561), num_quantizers=1)
        self.assertEqual(tail['gen_wav'].shape, (1, 2, 2561))
        self.assertEqual(tail['vq_code'].shape[-1], 3)
        mono = TrainableMossAudioTokenizer(channels=1)
        self.assertEqual(mono(torch.randn(1, 1, 1281), num_quantizers=1)['gen_wav'].shape, (1, 1, 1281))

    def test_normalized_vq_scale_and_decode(self):
        model = TrainableMossAudioTokenizer(l2_normalize=True, quantizer_dropout=0)
        # Perturb raw parameter magnitudes to catch search-only normalization.
        with torch.no_grad():
            for layer in model.codec.quantizer.quantizers:
                layer.codebook.weight.mul_(10000)
                layer.in_proj.bias.add_(1000)
        wav = torch.randn(1, 2, 2560)
        output = model(wav, num_quantizers=16)
        # Unit vectors: squared distance <= 4, mean over codebook_dim=8.
        self.assertTrue(torch.all(output['vq_loss'] <= 4 / 8 * (0.25 + 1.0) + 1e-5))
        output['vq_loss'].sum().backward()
        for layer in model.codec.quantizer.quantizers:
            self.assertTrue(torch.isfinite(layer.codebook.weight.grad).all())
            vectors = layer.embed_code(torch.arange(16))
            torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(16))
        model.eval()
        with torch.no_grad():
            hidden = model.codec.quantizer.decode_codes(output['vq_code'])
            lengths = torch.tensor([hidden.shape[-1]])
            for layer in model.codec.decoder:
                hidden, lengths = layer(hidden, lengths)
            decoded, _ = model.codec._restore_channels_from_codec(hidden, lengths)
        torch.testing.assert_close(output['gen_wav'], decoded, atol=2e-5, rtol=2e-5)

    def test_upstream_encode_decode_interface(self):
        model = TrainableMossAudioTokenizer(num_quantizers=2).eval().codec
        wav = torch.randn(1, 2, 2560)
        encoded = model.encode(wav, return_dict=True)
        self.assertEqual(encoded.audio_codes.shape, (2, 1, 2))
        decoded = model.decode(encoded.audio_codes, return_dict=True)
        self.assertEqual(decoded.audio.shape, wav.shape)
        decoded_prefix = model.decode(encoded.audio_codes[:1], return_dict=True)
        self.assertEqual(decoded_prefix.audio.shape, wav.shape)

    def test_training_checkpoint_resume_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Different source rates and channels exercise resampling/upmixing.
            sf.write(root / 'mono.wav', np.random.randn(7000).astype('float32') * .01, 22050)
            sf.write(root / 'stereo.wav', np.random.randn(6000, 2).astype('float32') * .01, 16000)
            manifest = root / 'train.txt'
            manifest.write_text(f'mono|{root / "mono.wav"}\nstereo|{root / "stereo.wav"}\n')
            with initialize(config_path='../config', version_base=None):
                cfg = compose(config_name='moss_16khz', overrides=[
                    f'dataset.train.filelist={manifest}', f'dataset.val.filelist={manifest}',
                    'dataset.train.batch_size=2', 'dataset.val.batch_size=2',
                    'dataset.num_workers=0', 'dataset.min_audio_length=4097',
                    'train.stages.pretrain_steps=2',
                    'model.moss_nano.num_quantizers=2', 'model.moss_nano.quantizer_dropout=0',
                    'model.mpd.periods=[2]', 'model.mpd.channels=2',
                    'model.mpd.max_downsample_channels=16', 'model.mstft.channels=2',
                    'model.mstft.max_downsample_channels=16',
                    'model.mstft.stft_params.fft_sizes=[128]',
                    'model.mstft.stft_params.hop_sizes=[32]',
                    'model.mstft.stft_params.win_lengths=[128]',
                ])
            data = DataModule(cfg)
            batch = next(iter(data.train_dataloader()))
            self.assertEqual(batch['wav'].shape, (2, 2, 4097))
            json_manifest = root / 'train.json'
            json_manifest.write_text(json.dumps([{'audio_filepath': str(root / 'mono.wav')}]))
            cfg.dataset.val.filelist = str(json_manifest)
            cfg.dataset.min_duration_sec = 0
            self.assertEqual(len(FSDataset('val', cfg)), 1)
            model = CodecLightningModule(cfg)
            trainer = pl.Trainer(accelerator='cpu', devices=1, precision=32,
                                 max_steps=2, logger=False, enable_checkpointing=False,
                                 enable_model_summary=False, num_sanity_val_steps=0,
                                 limit_val_batches=1, default_root_dir=directory)
            disc_before = {k: v.clone() for k, v in model.model['discriminator'].state_dict().items()}
            trainer.fit(model, datamodule=data)
            self.assertEqual(model.completed_batches, 2)
            self.assertFalse(trainer.optimizers[1].state)
            for key, value in model.model['discriminator'].state_dict().items():
                torch.testing.assert_close(value, disc_before[key], rtol=0, atol=0)
            self.assertEqual(trainer.global_step, 2)
            checkpoint = root / 'test.ckpt'
            trainer.save_checkpoint(checkpoint)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            restored = _build_model(saved['hyper_parameters']['cfg'], saved, 'cpu')
            model.eval()
            torch.testing.assert_close(model.inference(batch['wav']), restored.inference(batch['wav']))
            resumed = pl.Trainer(accelerator='cpu', devices=1, precision=32,
                                 max_steps=4, logger=False, enable_checkpointing=False,
                                 enable_model_summary=False, num_sanity_val_steps=0,
                                 limit_val_batches=0, default_root_dir=directory)
            resumed_model = CodecLightningModule(cfg)
            resumed.fit(resumed_model, datamodule=data, ckpt_path=checkpoint)
            self.assertEqual(resumed_model.completed_batches, 3)
            self.assertTrue(resumed.optimizers[1].state)
            self.assertEqual(resumed.callback_metrics['train_stage'].item(), 2)
            self.assertTrue(any(not torch.equal(value, disc_before[key])
                                for key, value in resumed_model.model['discriminator'].state_dict().items()))
            self.assertEqual(resumed.global_step, 4)

if __name__ == '__main__':
    unittest.main()
