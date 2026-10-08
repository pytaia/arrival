import os
import time
import random
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import ScoreDataset, collate_fn
from model import CRNN
from vocab import BLANK_ID, decode_ctc_greedy
from utils import ensure_dir

cfg = Config()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    seed = cfg.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def greedy_decode_batch(logits: torch.Tensor):
    pred = logits.argmax(dim=-1).transpose(0, 1)
    return [decode_ctc_greedy(row.tolist()) for row in pred]


def values_accuracy(pred_tokens_batch, gt_tokens_batch):
    correct = 0
    total = 0
    for pred, gt in zip(pred_tokens_batch, gt_tokens_batch):
        # exact token acc is still useful, but token-level rough metric is also okay
        n = min(len(pred), len(gt))
        for i in range(n):
            if pred[i] == gt[i]:
                correct += 1
        total += max(len(pred), len(gt))
    return correct, total


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, compiled_model=False):
    raw_model = model._orig_mod if compiled_model and hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "config": asdict(cfg),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, compiled_model=False):
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model._orig_mod if compiled_model and hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def make_loader(dataset, shuffle: bool, device: str):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(cfg.prefetch_factor if cfg.num_workers > 0 else None),
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
        drop_last=False,
    )


def main():
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = cfg.use_amp and device == "cuda"
    use_compile = cfg.use_compile and device == "cuda" and hasattr(torch, "compile")
    use_channels_last = cfg.use_channels_last and device == "cuda"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ensure_dir(cfg.checkpoint_dir)

    train_ds = ScoreDataset(cfg.train_dir, cfg.train_labels)
    val_ds = ScoreDataset(cfg.val_dir, cfg.val_labels)

    train_loader = make_loader(train_ds, True, device)
    val_loader = make_loader(val_ds, False, device)

    model = CRNN().to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    compiled_model = False
    if use_compile:
        model = torch.compile(model)
        compiled_model = True

    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    start_epoch = 0
    no_improve_epochs = 0

    last_ckpt = os.path.join(cfg.checkpoint_dir, "last.pt")
    best_ckpt = os.path.join(cfg.checkpoint_dir, "best.pt")

    if os.path.exists(last_ckpt):
        print(f"Resuming from {last_ckpt}")
        start_epoch, best_val = load_checkpoint(
            last_ckpt,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            compiled_model=compiled_model,
        )

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.perf_counter()

        model.train()
        train_loss = 0.0
        train_batches = 0

        train_bar = tqdm(train_loader, desc=f"train {epoch+1}/{cfg.epochs}")
        for xs, targets, target_lengths, _raw in train_bar:
            xs = xs.to(device, non_blocking=(device == "cuda"))
            targets = targets.to(device, non_blocking=(device == "cuda"))
            target_lengths = target_lengths.to(device, non_blocking=(device == "cuda"))

            if use_channels_last:
                xs = xs.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(xs)
                    log_probs = logits.log_softmax(dim=-1)
                    input_lengths = torch.full(
                        size=(xs.size(0),),
                        fill_value=logits.size(0),
                        dtype=torch.long,
                        device=device,
                    )
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xs)
                log_probs = logits.log_softmax(dim=-1)
                input_lengths = torch.full(
                    size=(xs.size(0),),
                    fill_value=logits.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            train_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss /= max(1, train_batches)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        exact = 0
        total = 0
        token_correct = 0
        token_total = 0

        do_exact = ((epoch + 1) % cfg.compute_exact_acc_every == 0)
        do_token = ((epoch + 1) % cfg.compute_token_acc_every == 0)

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="val")
            for xs, targets, target_lengths, raw in val_bar:
                xs = xs.to(device, non_blocking=(device == "cuda"))
                targets = targets.to(device, non_blocking=(device == "cuda"))
                target_lengths = target_lengths.to(device, non_blocking=(device == "cuda"))

                if use_channels_last:
                    xs = xs.contiguous(memory_format=torch.channels_last)

                if use_amp:
                    with torch.amp.autocast("cuda", enabled=True):
                        logits = model(xs)
                        log_probs = logits.log_softmax(dim=-1)
                        input_lengths = torch.full(
                            size=(xs.size(0),),
                            fill_value=logits.size(0),
                            dtype=torch.long,
                            device=device,
                        )
                        loss = criterion(log_probs, targets, input_lengths, target_lengths)
                else:
                    logits = model(xs)
                    log_probs = logits.log_softmax(dim=-1)
                    input_lengths = torch.full(
                        size=(xs.size(0),),
                        fill_value=logits.size(0),
                        dtype=torch.long,
                        device=device,
                    )
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                val_loss += loss.item()
                val_batches += 1

                if do_exact or do_token:
                    pred_tokens = greedy_decode_batch(logits.float().cpu())

                    if do_exact:
                        for pred, truth in zip(pred_tokens, raw):
                            if pred == truth:
                                exact += 1
                            total += 1

                    if do_token:
                        c, t = values_accuracy(pred_tokens, raw)
                        token_correct += c
                        token_total += t

        val_loss /= max(1, val_batches)
        exact_acc = exact / max(1, total) if do_exact else None
        token_acc = token_correct / max(1, token_total) if do_token else None

        scheduler.step(val_loss)

        msg = (
            f"epoch={epoch+1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} "
            f"time={time.perf_counter()-epoch_start:.1f}s"
        )
        if exact_acc is not None:
            msg += f" exact_seq_acc={exact_acc:.4f}"
        if token_acc is not None:
            msg += f" token_acc={token_acc:.4f}"
        print(msg)

        if (epoch + 1) % cfg.save_last_every == 0:
            save_checkpoint(
                last_ckpt, model, optimizer, scheduler, scaler,
                epoch + 1, best_val, compiled_model=compiled_model
            )

        if val_loss < best_val:
            best_val = val_loss
            no_improve_epochs = 0
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler, scaler,
                epoch + 1, best_val, compiled_model=compiled_model
            )
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("done")


if __name__ == "__main__":
    main()
