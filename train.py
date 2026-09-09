import pytorch_lightning as pl
import hydra
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from data_module import DataModule
from lightning_module import CodecLightningModule
from common.lightning_compat import compatible_precision

seed = 1024
seed_everything(seed)

@hydra.main(config_path='config', config_name='moss_16khz', version_base=None)
def train(cfg):
    trainer_options = dict(cfg.train.trainer)
    if trainer_options.get('max_steps') is None:
        stages = cfg.train.stages
        pretrain_steps, adversarial_steps = int(stages.pretrain_steps), int(stages.adversarial_steps)
        if pretrain_steps < 0 or adversarial_steps < 0 or pretrain_steps + adversarial_steps == 0:
            raise ValueError('Stage lengths must be nonnegative, with at least one training batch.')
        trainer_options['max_steps'] = pretrain_steps + 2 * adversarial_steps
    requested_precision = trainer_options.get('precision', 32)
    trainer_options['precision'] = compatible_precision(requested_precision, pl.__version__)
    if trainer_options['precision'] != requested_precision:
        print(f"Lightning {pl.__version__}: precision {requested_precision!r} -> {trainer_options['precision']!r}")
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir,
                            save_top_k=1, save_last=True,
                            every_n_train_steps=cfg.train.get('checkpoint_interval', 500), monitor='mel_loss', mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)

    devices = cfg.train.trainer.devices
    multi_device = (isinstance(devices, int) and devices > 1) or (not isinstance(devices, (str, int)) and len(devices) > 1)
    if cfg.train.trainer.get('accumulate_grad_batches', 1) != 1:
        raise ValueError('Manual GAN optimization requires accumulate_grad_batches=1.')
    # Omit strategy for a single device: Lightning 1.x does not register 'auto'.
    if multi_device:
        trainer_options['strategy'] = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        **trainer_options,
        callbacks=callbacks,
        default_root_dir=cfg.log_dir,
        limit_train_batches=1.0 if not cfg.debug else 0.001
    )

    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=cfg.get("resume_ckpt", None),
    )

    print(f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}')

if __name__ == '__main__':
    train()
