import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from scripts.utils.config import load_yaml, deep_update
from scripts.utils.seed import seed_everything
from scripts.utils.logging import get_rich_logger
from scripts.utils.instantiate import make_run_dir, instantiate
import torch
torch.set_float32_matmul_precision("high")

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--config", required=True, help="experiment yaml")
    fit.add_argument("--default", default="configs/default.yaml", help="base default yaml")

    test = sub.add_parser("test")
    test.add_argument("--config", required=True, help="experiment yaml")
    test.add_argument("--default", default="configs/default.yaml", help="base default yaml")
    test.add_argument("--ckpt", required=True, help="checkpoint path (.ckpt)")

    return p.parse_args()


# def build_callbacks(run_dir: Path):
#     ckpt_dir = run_dir / "checkpoints"
#     ckpt_dir.mkdir(parents=True, exist_ok=True)

#     checkpoint = ModelCheckpoint(
#         dirpath=str(ckpt_dir),
#         filename="{epoch:03d}-{val_acc:.4f}",
#         monitor="val/acc",
#         mode="max",
#         save_top_k=1,
#         save_last=True,
#     )
#     lr_monitor = LearningRateMonitor(logging_interval="step")
#     return [checkpoint, lr_monitor]


def build_callbacks(run_dir: Path):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch:03d}-{val_loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return [checkpoint, lr_monitor]




def main():
    args = parse_args()

    base_cfg = load_yaml(args.default)
    exp_cfg = load_yaml(args.config)
    cfg = deep_update(base_cfg, exp_cfg)

    # seed
    seed_everything(cfg.get("seed", 42), deterministic=cfg["trainer"].get("deterministic", True))

    # run dir + logger
    run_dir = make_run_dir(cfg["experiment"]["output_dir"], cfg["experiment"]["name"])
    log = get_rich_logger(run_dir)
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config merged: default={args.default} exp={args.config}")

    # datamodule + model
    dm = instantiate(cfg["data"]["target"], cfg["data"].get("params"))
    
    model = instantiate(cfg["model"]["target"], cfg["model"].get("params"))
    
    
    # tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")

    # trainer
    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        logger=tb_logger,
        callbacks=build_callbacks(run_dir),
        log_every_n_steps=cfg["experiment"].get("log_every_n_steps", 50),
        **cfg["trainer"],
    )

    if args.cmd == "fit":
        trainer.fit(model, datamodule=dm)
        log.info("Fit done.")
        log.info(f"Best ckpt: {trainer.checkpoint_callback.best_model_path}")

    elif args.cmd == "test":
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt)
        log.info("Test done.")


if __name__ == "__main__":
    main()
