from pathlib import Path
import logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration  # optional

def log_metric(logger, name, value, epoch, global_step, log_wandb):
    logger.info(f"[epoch {epoch}, step {global_step}] {name}: {value:.4f}")
    if log_wandb:
        logger.wandb_log({name: value}, step=global_step, epoch=epoch)

def create_logger(
    accelerator: Accelerator,
    logging_dir: str | Path | None = None,
    *,
    all_processes: bool = False,
    log_level: str = "INFO",
    enable_wandb: bool = False,
    wandb_init_kwargs: dict | None = None,
):
    """
    Return an Accelerate‑compatible logger.

    * `all_processes=False`  ➜ log only on the main process (default).
    * `logging_dir=None`     ➜ skip file logging.
    * `enable_wandb=True`    ➜ initialise W&B through Accelerate’s tracker API.
    """
    # Accelerate logger adapter
    logger = get_logger(__name__, log_level=log_level)

    # Add file handler to the base logger (not the adapter)
    if logging_dir and (all_processes or accelerator.is_main_process):
        logging_dir = Path(logging_dir)
        logging_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logging_dir / f"{wandb_init_kwargs['name']}.txt", mode="a")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.logger.addHandler(fh)  # <- FIXED HERE

    # 3) WandB via Accelerate’s tracker interface
    logger.wandb_log = lambda *args, **kwargs: None  # no‑op unless enabled

    if enable_wandb and accelerator.is_main_process:
        # Make sure the accelerator knows we want WandB
        if accelerator.state.log_with is None:
            accelerator.state.log_with = ["wandb"]

        accelerator.init_trackers(
            project_name=wandb_init_kwargs.pop("project"),
            config=wandb_init_kwargs.pop("config", None),
            init_kwargs={"wandb": wandb_init_kwargs},
        )

        # Thin wrapper so old `logger.wandb_log` calls still work
        logger.wandb_log = lambda metrics, step=None: accelerator.log(
            metrics if isinstance(metrics, dict) else {"value": metrics},
            step=step,
        )

    return logger
