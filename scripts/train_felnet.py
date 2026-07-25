"""Train FELNet (paper Algorithm 2, Sec. 2.5).

Uses the episode sampler (paper's dense random sampling: N_o frames at interval
k, N_s SSIs each), the three RMSE losses (overlap / confidence / embedding),
and Adam + cosine-annealing LR as the paper specifies.

An "epoch" here is --steps-per-epoch optimizer steps of randomly sampled
episodes (episode sampling is stochastic, not a fixed pass over a dataset).

Smoke test / timing (few steps, prints ms/step):
    python scripts/train_felnet.py --epochs 1 --steps-per-epoch 20 --name smoke --time-only

Full run:
    python scripts/train_felnet.py --epochs 50 --steps-per-epoch 500 --name full_run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jmdst.data.felnet_episode import FELNetEpisodeDataset, collate_felnet_episodes
from jmdst.models import FELNet, FELNetConfig
from jmdst.training import FELNetLoss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified-root", default="data/unified")
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset (visdrone/uavdt). Default: both.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--episodes-per-step", type=int, default=2, help="Episodes merged per optimizer step (batch dim).")
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--n-o", type=int, default=8, help="Frames per episode (paper N_o).")
    parser.add_argument("--n-s", type=int, default=8, help="SSIs per frame (paper N_s).")
    parser.add_argument("--num-positive", type=int, default=128)
    parser.add_argument("--num-negative", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="outputs/felnet_runs")
    parser.add_argument("--name", default="train")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--time-only", action="store_true", help="Run the steps, print timing, do not save checkpoints.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = FELNetEpisodeDataset(
        unified_root=args.unified_root,
        dataset=args.dataset,
        split=args.split,
        k_max=args.k_max,
        n_o=args.n_o,
        n_s=args.n_s,
        length=args.steps_per_epoch * args.episodes_per_step,
        horizontal_flip=True,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset.sequences)} sequences (dataset={args.dataset or 'all'}, split={args.split}).")

    loader = DataLoader(
        dataset,
        batch_size=args.episodes_per_step,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_felnet_episodes,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = FELNet(FELNetConfig(embedding_dim=args.embedding_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = FELNetLoss(num_positive=args.num_positive, num_negative=args.num_negative)

    save_dir = Path(args.project).resolve() / args.name
    if not args.time_only:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"device={device} total_steps={total_steps} "
          f"episode_size~={args.n_o * args.n_s} SSIs x {args.episodes_per_step} episodes/step")

    model.train()
    global_step = 0
    step_times: list[float] = []
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for batch in loader:
            t0 = time.perf_counter()
            for key in ("ssi", "overlaps", "confidences", "center_anchor_index", "identity"):
                batch[key] = batch[key].to(device, non_blocking=True)

            outputs = model(batch["ssi"])
            losses = criterion(outputs, batch)

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            scheduler.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            epoch_losses.append(float(losses["total"].item()))
            global_step += 1

            if global_step % args.log_every == 0:
                recent = sum(step_times[-args.log_every:]) / min(len(step_times), args.log_every)
                print(
                    f"epoch {epoch}/{args.epochs} step {global_step}/{total_steps} "
                    f"L={losses['total'].item():.4f} "
                    f"(Lo={losses['overlap'].item():.4f} Lc={losses['confidence'].item():.4f} "
                    f"Le={losses['embedding'].item():.4f}) "
                    f"lr={scheduler.get_last_lr()[0]:.2e} {recent*1000:.0f} ms/step"
                )

        mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        history.append({"epoch": epoch, "mean_total_loss": mean_loss})
        print(f"== epoch {epoch} mean total loss: {mean_loss:.4f} ==")

        if not args.time_only:
            ckpt = {
                "model": model.state_dict(),
                "config": vars(model.config),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, save_dir / "last.pt")
            if mean_loss <= min(h["mean_total_loss"] for h in history):
                torch.save(ckpt, save_dir / "best.pt")

    median_ms = sorted(step_times)[len(step_times) // 2] * 1000
    print(f"\nTiming: {len(step_times)} steps, median {median_ms:.0f} ms/step.")
    if not args.time_only:
        (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Checkpoints + history saved to {save_dir}")


if __name__ == "__main__":
    main()
