import logging
from pathlib import Path

from rich.logging import RichHandler


def get_rich_logger(run_dir: Path) -> logging.Logger:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ltproj")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    # console
    ch = RichHandler(rich_tracebacks=True, markup=True)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
