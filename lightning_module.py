import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from moss_trainable import TrainableMossAudioTokenizer
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss
from common.schedulers import WarmupLR

class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        print(cfg)
        self.cfg = cfg
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.completed_batches = 0
        self.pretrain_steps = int(cfg.train.get("stages", {}).get("pretrain_steps", 0))
        if self.pretrain_steps < 0:
            raise ValueError("pretrain_steps must be nonnegative")

    def construct_model(self):
        generator = TrainableMossAudioTokenizer(**self.cfg.model.moss_nano)
        if self.cfg.preprocess.audio.sr != generator.codec.sampling_rate:
            raise ValueError('Dataset and model sample rates must match.')
        if self.cfg.preprocess.audio.channels != generator.codec.number_channels:
            raise ValueError('Dataset and model channel counts must match.')
        mpdcfg = self.cfg.model.mpd
        mpd = HiFiGANMultiPeriodDiscriminator(
                    periods=mpdcfg.periods,
                    max_downsample_channels=mpdcfg.max_downsample_channels,
                    channels=mpdcfg.channels,
                    channel_increasing_factor=mpdcfg.channel_increasing_factor,
                )
        mstftcfg = self.cfg.model.mstft
        mstft = SpecDiscriminator(
                    stft_params=mstftcfg.stft_params,
                    in_channels=mstftcfg.in_channels,
                    out_channels=mstftcfg.out_channels,
                    kernel_sizes=mstftcfg.kernel_sizes,
                    channels=mstftcfg.channels,
                    max_downsample_channels=mstftcfg.max_downsample_channels,
                    downsample_scales=mstftcfg.downsample_scales,
                    use_weight_norm=mstftcfg.use_weight_norm,
                )
        model = nn.ModuleDict({
                    'generator': generator,
                    'discriminator': mpd,
                    'spec_discriminator': mstft,
                })
        self.model = model

    def construct_criteria(self):
        cfg = self.cfg.train
        criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_feat_match_loss:
            criteria['fm_loss'] = nn.L1Loss()
        criteria['gan_loss'] = GANLoss()
        self.criteria = criteria
        print(criteria)

    def forward(self, batch):
        wav = batch['wav']
        output = self.model['generator'](wav, batch.get('lengths'))
        self.log('codebook_utilization', output['utilization'].mean(), on_step=True, on_epoch=True)
        return output

    @torch.inference_mode()
    def inference(self, wav, lengths=None, num_quantizers=None):
        return self.model['generator'].inference(wav, lengths, num_quantizers)

    def compute_disc_loss(self, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        # Reuse the existing mono losses/discriminators on each stereo channel.
        y, y_ = y.reshape(-1, 1, y.shape[-1]), y_.reshape(-1, 1, y_.shape[-1])
        y_ = y_.detach()
        p = self.model['discriminator'](y)
        p_ = self.model['discriminator'](y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        sd_p = self.model['spec_discriminator'](y)
        sd_p_ = self.model['spec_discriminator'](y_)
        for real_features, fake_features in zip(sd_p, sd_p_):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(
                real_features[-1], fake_features[-1]
            )
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output

    def compute_gen_loss(self, output, adversarial=True):
        y, y_ = output['gt_wav'], output['gen_wav']
        # Reuse the existing mono losses/discriminators on each stereo channel.
        y, y_ = y.reshape(-1, 1, y.shape[-1]), y_.reshape(-1, 1, y_.shape[-1])
        vq_loss, vq_code = output['vq_loss'], output['vq_code']
        gen_loss = .0
        self.set_discriminator_gradients(False)
        output = {}
        cfg = self.cfg.train

        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output['mel_loss'] = mel_loss

        if adversarial:
            # gan loss
            p_ = self.model['discriminator'](y_)
            adv_loss_list = []
            for i in range(len(p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
            sd_p_ = self.model['spec_discriminator'](y_)
            for features in sd_p_:
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(features[-1]))
            adv_loss = sum(adv_loss_list)
            gen_loss += adv_loss * cfg.lambdas.lambda_adv
            output['adv_loss'] = adv_loss

            # fm loss
            if cfg.use_feat_match_loss:
                fm_loss = 0.
                with torch.no_grad():
                    p = self.model['discriminator'](y)
                for i in range(len(p_)):
                    for j in range(len(p_[i]) - 1):
                        fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
                gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
                output['fm_loss'] = fm_loss
                spec_fm_loss = 0.
                with torch.no_grad():
                    sd_p = self.model['spec_discriminator'](y)
                for fake_features, real_features in zip(sd_p_, sd_p):
                    for fake_feature, real_feature in zip(fake_features[:-1], real_features[:-1]):
                        spec_fm_loss += self.criteria['fm_loss'](
                            fake_feature, real_feature.detach()
                        )
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output['spec_fm_loss'] = spec_fm_loss

        # vq
        if vq_loss is not None:
            vq_loss = sum(vq_loss)
            gen_loss += vq_loss
            output['vq_loss'] = vq_loss

        output['gen_loss'] = gen_loss
        return output


    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('active_rq_step', output['vq_code'].shape[0],
                 on_step=True, on_epoch=False, prog_bar=True, logger=True,
                 sync_dist=False, batch_size=batch['wav'].shape[0])

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        adversarial = self.completed_batches >= self.pretrain_steps
        self.log("train_stage", 2 if adversarial else 1, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_batches", self.completed_batches, on_step=True, on_epoch=False)
        disc_losses = {}
        # No discriminator forwards, optimizer or scheduler steps in stage 1.
        if adversarial:
            self.set_discriminator_gradients(True)
            disc_losses = self.compute_disc_loss(output)
            disc_loss = disc_losses['disc_loss']
            disc_opt.zero_grad()
            self.manual_backward(disc_loss)
            self.clip_gradients(disc_opt, gradient_clip_val=self.cfg.train.disc_grad_clip, gradient_clip_algorithm='norm')
            disc_opt.step()
            disc_sche.step()

        # generator
        gen_losses = self.compute_gen_loss(output, adversarial=adversarial)
        gen_loss = gen_losses['gen_loss']
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(gen_opt, gradient_clip_val=self.cfg.train.gen_grad_clip, gradient_clip_algorithm='norm')
        gen_opt.step()
        gen_sche.step()
        self.set_discriminator_gradients(True)

        self.completed_batches += 1
        self.log_dict(disc_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)
        self.log_dict(gen_losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size, sync_dist=True)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['codec_training_progress'] = {
            'completed_batches': self.completed_batches,
            'pretrain_steps': self.pretrain_steps,
        }

    def on_load_checkpoint(self, checkpoint):
        progress = checkpoint.get('codec_training_progress')
        if progress is not None:
            if int(progress['pretrain_steps']) != self.pretrain_steps:
                raise ValueError('Resume requires the original train.stages.pretrain_steps setting.')
            self.completed_batches = int(progress['completed_batches'])
        else:
            # Older checkpoints were trained with GAN from the first batch.
            import warnings
            warnings.warn('Legacy GAN checkpoint: resuming in stage 2 with pretraining treated as complete.')
            self.completed_batches = self.pretrain_steps + int(checkpoint.get('global_step', 0)) // 2

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        y, pred = output['gt_wav'], output['gen_wav']
        if 'mel_loss' in self.criteria:
            loss = self.criteria['mel_loss'](pred.reshape(-1, pred.shape[-1]), y.reshape(-1, y.shape[-1]))
            self.log('val_mel_loss', loss, sync_dist=True, batch_size=y.shape[0])

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        from itertools import chain
        disc_params = chain(self.model['discriminator'].parameters(),
                            self.model['spec_discriminator'].parameters())
        gen_params = self.model['generator'].parameters()

        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)
        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')
        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.model['discriminator'].parameters():
            p.requires_grad = flag

        for p in self.model['spec_discriminator'].parameters():
            p.requires_grad = flag
