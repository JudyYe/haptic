import os
import signal
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

import haptic.models.haptic as haptic_module
from haptic.configs import dataset_config
from haptic.datasets import HAPTICDataModule
from haptic.utils.misc import log_hyperparameters, task_wrapper
from haptic.utils.pylogger import get_pylogger
from nnutils import model_utils
from nnutils.logger import LoggerCallback, build_logger

signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)


@hydra.main("haptic/configs_hydra", "train", version_base=None)
# @task_wrapper
def main(cfg: DictConfig) -> Optional[float]:
    # Load dataset config
    dataset_cfg = dataset_config()

    # Save configs
    model_utils.save_configs(cfg, dataset_cfg, cfg.paths.output_dir)

    # Setup training and validation datasets
    datamodule = HAPTICDataModule(cfg, dataset_cfg)

    OmegaConf.save(config=cfg, f=os.path.join(cfg.exp_dir, "config.yaml"))
    # Setup model
    class_ = getattr(haptic_module, cfg.MODEL.get("TARGET", "haptic"))
    model = class_(cfg)
    # model = haptic(cfg)
    if cfg.ckpt_path is not None:
        log.info(f"Restoring model from checkpoint: {cfg.ckpt_path}")
        haptic_weight = torch.load(cfg.ckpt_path)["state_dict"]
        miss_keys, _, _ = model_utils.load_my_state_dict(model, haptic_weight)
        model.new_keys = miss_keys
    logger = build_logger(cfg.expname, cfg.exp_dir, log=cfg.log)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, "checkpoints"),
        every_n_train_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint_callback,
        lr_monitor,
        LoggerCallback(),
    ]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=loggers,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path="last")
    log.info("Fitting done")


if __name__ == "__main__":
    main()
