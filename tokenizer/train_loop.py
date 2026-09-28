# train.py
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from itertools import cycle
import time

from tokenizer.core import BaseTaskModel
from tokenizer.utils.logger import create_logger, log_metric
from tokenizer.utils.checkpointing import save_checkpoint, load_checkpoint
from tokenizer.utils.plot_validation import save_contourf_comparison
from tokenizer.utils.normalization import denormalize_tensor
from tokenizer.utils.error_metrics import (
    rel_l1_error_tensor,
    rel_l2_error_tensor,
    top_percent_abs_rel_l1_error,
    gradient_top_percent_abs_rel_l1_error,
    tail_conditional_rel_l1_error,
    tail_conditional_rel_l2_error,
)

def validate(
    task: BaseTaskModel,
    val_loader: DataLoader,
    global_step: int,
    ):
    task.eval()
    l1_accum_loss = 0
    usage_accum = 0
    metric_sums = {
        "physical_rel_l1_pct": 0.0,
        "physical_rel_l2_pct": 0.0,
        "physical_top1_abs_rel_l1_pct": 0.0,
        "physical_grad_top1_abs_rel_l1_pct": 0.0,
        "physical_tail95_rel_l1_pct": 0.0,
        "physical_tail95_rel_l2_pct": 0.0,
    }
    start_time = time.time()
    with torch.no_grad():
        for v_step, v_batch in enumerate(val_loader):

            # # imagenet hack -------------------------------------------------------
            # device = next(task.model.parameters()).device 
            # v_batch["field_variables_in"] = v_batch["field_variables_in"].to(device)
            # v_batch["field_variables_out"] = v_batch["field_variables_out"].to(device)
            # v_batch["field_variables_in_mean"] = v_batch["field_variables_in_mean"].to(device)
            # v_batch["field_variables_in_std"] = v_batch["field_variables_in_std"].to(device)
            # # ----------------------------------------------------------------------
            recon, l1_loss, usage_pct = task(v_batch, mode="val", global_step=global_step)
            l1_accum_loss += l1_loss
            usage_accum += usage_pct

            target_norm = v_batch["field_variables_out"]
            mean = v_batch["field_variables_in_mean"]
            std = v_batch["field_variables_in_std"]

            target_phys = denormalize_tensor(target_norm, mean, std)
            recon_phys = denormalize_tensor(recon, mean, std)

            metric_sums["physical_rel_l1_pct"] += rel_l1_error_tensor(recon_phys, target_phys).item()
            metric_sums["physical_rel_l2_pct"] += rel_l2_error_tensor(recon_phys, target_phys).item()
            metric_sums["physical_top1_abs_rel_l1_pct"] += top_percent_abs_rel_l1_error(
                recon_phys, target_phys, top_percent=1.0
            ).item()
            metric_sums["physical_grad_top1_abs_rel_l1_pct"] += gradient_top_percent_abs_rel_l1_error(
                recon_phys, target_phys, top_percent=1.0
            ).item()
            metric_sums["physical_tail95_rel_l1_pct"] += tail_conditional_rel_l1_error(
                recon_phys, target_phys, tail_quantile=0.95
            ).item()
            metric_sums["physical_tail95_rel_l2_pct"] += tail_conditional_rel_l2_error(
                recon_phys, target_phys, tail_quantile=0.95
            ).item()

            if v_step == 9: 
                break
    end_time = time.time()
    # print(f"Validation took {end_time - start_time:.2f} seconds for {v_step+1} steps: {(end_time - start_time)/(v_step+1):.2f} seconds/step.")

    avg_usage = usage_accum / (v_step + 1)
    avg_metrics = {k: v / (v_step + 1) for k, v in metric_sums.items()}
    # return the last v_batch and reconstructions to plot them for visiual updates during training
    return l1_accum_loss / (v_step+1), v_batch["field_variables_out"], recon, avg_usage, avg_metrics 
    

def fit(
    task: BaseTaskModel,
    model_config,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
):
    train_config = model_config.training_hyperparameters
    epochs = train_config.epochs
    log_every = train_config.log_every
    val_every = train_config.val_every
    logging_dir = train_config.logging_dir
    checkpoint_dir = train_config.checkpoint_dir
    enable_wandb = train_config.enable_wandb
    experiment_name = train_config.experiment_name
    checkpoint = train_config.load_from_checkpoint
    max_steps = int(getattr(train_config, "max_steps", 0) or 0)   # 0 = no cap

    # Prepare accelerator
    accelerator = Accelerator(
        mixed_precision="bf16" if train_config.fp16 else "no",
        gradient_accumulation_steps=train_config.grad_accum_steps,
    )

    optimizers, schedulers = task.configure_optimizers()  # Unpack both
    objs = accelerator.prepare(*optimizers, train_loader,
                           *(val_loader,) if val_loader is not None else ())
    

    task.prepare_model(accelerator)

    n_opt = len(optimizers)               # 1‑or‑2 in your case
    optimizers     = list(objs[0: 0+n_opt])
    assert(n_opt==len(optimizers))
    train_loader   = objs[0+n_opt]
    if val_loader is not None:
        val_loader = objs[1 + n_opt]


    # Initialize logger and track first info
    logger = create_logger(
            accelerator,
            logging_dir=logging_dir,
            enable_wandb=enable_wandb,
            wandb_init_kwargs=dict(
                project="Phaedra_Project",
                name=f"Exp-{experiment_name}",
                entity="Phaedra",
            ),
    )

    # Load from checkpoint
    global_step = 0
    if checkpoint is not None:
        checkpoint_name = f"{experiment_name}_{checkpoint}"
        load_checkpoint(accelerator, checkpoint_dir, checkpoint_name, task.ema)
        global_step = checkpoint
        logger.info(f"🚀 Training restarts from checkpoint {checkpoint}!")
    else: 
        logger.info("🚀 Training starts!")

    logger.info(model_config)
    logger.info(f"Model Parameters: {sum(p.numel() for p in task.parameters()):,}")

    for epoch in range(epochs):
        task.train()
        running_loss = 0
        start_time = time.time()  # Start timing
        for step, batch in enumerate(train_loader):
            for optimizer_idx, optimizer in enumerate(optimizers):
                with accelerator.accumulate(task):
                    # # imagenet hack -------------------------------------------------------
                    # # 1. Identify the device your model is on
                    # device = next(task.model.parameters()).device 
                    # # 2. In your loop, move the input tensor to that device
                    # batch["field_variables_in"] = batch["field_variables_in"].to(device)
                    # batch["field_variables_out"] = batch["field_variables_out"].to(device)
                    # # -----------------------------------------------------------------------

                    loss = task(batch, 
                                mode="train",
                                global_step=global_step,
                                optimizer_idx=optimizer_idx)
                    if isinstance(loss, tuple):
                        loss, losses = loss
                    else:
                        losses = {"loss": loss}

                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        raise RuntimeError(f"[Rank {accelerator.process_index}] Loss is NaN or inf: {loss}")

                    torch.nn.utils.clip_grad_norm_(task.parameters(), max_norm=5.0)
                    accelerator.backward(loss)
                    optimizer.step()
                    task.ema.update()
                    optimizer.zero_grad()

                    running_loss += loss.item()
            
            # Do a validation step and step the schedulers
            if global_step % val_every == 0:
                task.eval()
                val_loss, x_validation, recon_validation, avg_usage, avg_metrics = validate(task, val_loader, global_step=global_step)

                # must convert val_loss to a tensor in order to reduce across ranks
                val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
                val_loss_reduced = accelerator.reduce(val_loss_tensor, reduction="mean")
                usage_tensor = torch.tensor(avg_usage, device=accelerator.device)
                usage_reduced = accelerator.reduce(usage_tensor, reduction="mean")
                reduced_metrics = {}
                for name, value in avg_metrics.items():
                    value_tensor = torch.tensor(value, device=accelerator.device)
                    reduced_metrics[name] = accelerator.reduce(value_tensor, reduction="mean")
                
                # Step the scheduler based on validation loss
                if accelerator.is_main_process:
                    for scheduler in schedulers:
                        # Log current learning rate
                        current_lr = optimizers[0].param_groups[0]['lr']
                        log_metric(logger, "learning_rate", current_lr, epoch, global_step, log_wandb=enable_wandb)
                        accelerator.print(f"Step {global_step} \t Current LR: {current_lr:.2e} \t Validation L1 Loss: {val_loss_reduced:.4f} \t Token Usage: {usage_reduced:.2f}%")
                        log_metric(logger, "validation L1 loss", val_loss_reduced, epoch, global_step, log_wandb=enable_wandb)
                        log_metric(logger, "validation token usage (%)", usage_reduced, epoch, global_step, log_wandb=enable_wandb)
                        for metric_name, metric_value in reduced_metrics.items():
                            log_metric(logger, metric_name, metric_value, epoch, global_step, log_wandb=enable_wandb)
                

            # Log training and save checkpoints
            if accelerator.is_main_process and global_step % log_every == 0 and step > 0:
                elapsed_time = time.time() - start_time
                steps_per_second = log_every / elapsed_time
                accelerator.print(f"Steps/sec: {steps_per_second:.2f}")
                log_metric(logger, "steps_per_second", steps_per_second, epoch, global_step, log_wandb=enable_wandb)

                value = running_loss / step if step != 0 else 0.0
                log_metric(logger, "training loss", value, epoch, global_step, log_wandb=enable_wandb)
                for key, val in losses.items():
                    log_metric(logger, key, val, epoch, global_step, log_wandb=enable_wandb)
                    accelerator.print(f"{key}: {val:.4f}")
                save_checkpoint(accelerator,global_step,checkpoint_dir = checkpoint_dir,name = f"{experiment_name}_{global_step}", ema=task.ema)


                start_time = time.time()
            # ── validation ──────────────────────────────────────────────────────
            if val_loader is not None and global_step % log_every==0 and global_step > -1:
                task.eval()
                val_loss, x_validation, recon_validation, avg_usage, avg_metrics = validate(task, val_loader, global_step=global_step)
                # must convert val_loss to a tensor in order to reduce across ranks
                val_loss_tensor = torch.tensor(val_loss, device=accelerator.device)
                val_loss_reduced = accelerator.reduce(val_loss_tensor, reduction="mean")
                
                # Reduce usage across ranks
                usage_tensor = torch.tensor(avg_usage, device=accelerator.device)
                usage_reduced = accelerator.reduce(usage_tensor, reduction="mean")
                reduced_metrics = {}
                for name, value in avg_metrics.items():
                    value_tensor = torch.tensor(value, device=accelerator.device)
                    reduced_metrics[name] = accelerator.reduce(value_tensor, reduction="mean")
                
                if accelerator.is_main_process:
                    log_metric(logger, "validation L1 loss", val_loss_reduced, epoch, global_step, log_wandb=enable_wandb)
                    log_metric(logger, "validation token usage (%)", usage_reduced, epoch, global_step, log_wandb=enable_wandb)
                    for metric_name, metric_value in reduced_metrics.items():
                        log_metric(logger, metric_name, metric_value, epoch, global_step, log_wandb=enable_wandb)

                    if isinstance(recon_validation, tuple):
                        diffusion_outputs, deterministic_outputs = recon_validation
                        save_contourf_comparison(x_validation, diffusion_outputs, f"{checkpoint_dir}/{experiment_name}_{global_step}/diffusion_samples.png")
                        save_contourf_comparison(x_validation, deterministic_outputs, f"{checkpoint_dir}/{experiment_name}_{global_step}/decoder_samples.png")
                    else:
                        # imagenet hack -------------------------------------------------------
                        x_validation = x_validation.to(torch.float32)
                        recon_validation = recon_validation.to(torch.float32)
                        # ----------------------------------------------------------------------
                        save_contourf_comparison(x_validation, recon_validation, f"{checkpoint_dir}/{experiment_name}_{global_step}/validation_samples.png")
                task.train()
                
            # update global step
            global_step += 1
            if max_steps and global_step >= max_steps:
                if accelerator.is_main_process:
                    save_checkpoint(accelerator, global_step, checkpoint_dir=checkpoint_dir,
                                    name=f"{experiment_name}_{global_step}", ema=task.ema)
                accelerator.wait_for_everyone()
                accelerator.print(f"[train] reached max_steps={max_steps}, stopping")
                return

        # Step the scheduler once per epoch
        for scheduler in schedulers:
            scheduler.step()
    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, global_step, checkpoint_dir=checkpoint_dir,
                    name=f"{experiment_name}_{global_step}", ema=task.ema)
    accelerator.print(f"[train] done after {global_step} steps")


