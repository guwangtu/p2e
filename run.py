import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from scripts.utils.config import load_yaml, deep_update
from scripts.utils.seed import seed_everything
from scripts.utils.logging import get_rich_logger
from scripts.utils.instantiate import make_run_dir, instantiate
import torch
torch.set_float32_matmul_precision("high")


class PeriodicSaveCallback(Callback):
    """Save checkpoint every N epochs for later evaluation."""

    def __init__(self, every_n_epochs: int = 50, save_dir: str = "checkpoints"):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0 and epoch > 0:
            ckpt_path = Path(self.save_dir) / f"periodic_epoch{epoch:03d}.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            print(f"\n[PeriodicSave] Saved checkpoint at epoch {epoch}: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--config", required=True, help="experiment yaml")
    fit.add_argument("--default", default="configs/default.yaml", help="base default yaml")
    fit.add_argument("--ckpt", default=None, help="resume from checkpoint (.ckpt)")

    test = sub.add_parser("test")
    test.add_argument("--config", required=True, help="experiment yaml")
    test.add_argument("--default", default="configs/default.yaml", help="base default yaml")
    test.add_argument("--ckpt", required=True, help="checkpoint path (.ckpt)")

    return p.parse_args()


def build_callbacks(run_dir: Path, test_every: int = 50):
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
    early_stop = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=15,
        verbose=True,
    )
    periodic_save = PeriodicSaveCallback(every_n_epochs=test_every, save_dir=str(ckpt_dir))
    return [checkpoint, lr_monitor, early_stop, periodic_save]


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

    # test_every from config (default 50)
    test_every = cfg["experiment"].get("test_every_n_epochs", 50)

    # trainer
    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        logger=tb_logger,
        callbacks=build_callbacks(run_dir, test_every=test_every),
        log_every_n_steps=cfg["experiment"].get("log_every_n_steps", 50),
        **cfg["trainer"],
    )

    if args.cmd == "fit":
        ckpt_path = args.ckpt if args.ckpt else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        log.info("Fit done.")
        log.info(f"Best ckpt: {trainer.checkpoint_callback.best_model_path}")

        # Auto-run test on periodic checkpoints and best checkpoint
        ckpt_dir = run_dir / "checkpoints"
        periodic_ckpts = sorted(ckpt_dir.glob("periodic_epoch*.ckpt"))
        best_ckpt = trainer.checkpoint_callback.best_model_path

        all_test_ckpts = [(str(p), p.stem) for p in periodic_ckpts]
        if best_ckpt:
            all_test_ckpts.append((best_ckpt, "best"))

        for ckpt_path_str, label in all_test_ckpts:
            log.info(f"Testing [{label}]: {ckpt_path_str}")
            trainer.test(model, datamodule=dm, ckpt_path=ckpt_path_str)
        log.info("All tests done.")

    elif args.cmd == "test":
        trainer.test(model, datamodule=dm, ckpt_path=args.ckpt)
        log.info("Test done.")


if __name__ == "__main__":
    main()
