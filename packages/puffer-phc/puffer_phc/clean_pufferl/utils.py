import os
import random

import torch
import numpy as np

from puffer_phc.config import TrainConfig

from dataclasses import asdict

import rich
from rich.console import Console
from rich.table import Table

import numpy as np


def save_checkpoint(uncompiled_policy, optimizer, train_cfg: TrainConfig, exp_id: str, epoch: int, global_step: int):
    path = os.path.join(train_cfg.data_dir, exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f"model_{epoch:06d}.pt"
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    checkpoint = {"config": asdict(train_cfg), "state_dict": uncompiled_policy.state_dict()}
    torch.save(checkpoint, model_path)

    checkpoint_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "agent_step": global_step,
        "update": epoch,
        "model_name": model_name,
        "exp_id": exp_id,
    }
    state_path = os.path.join(path, "trainer_state.pt")
    torch.save(checkpoint_state, state_path + ".tmp")
    os.rename(state_path + ".tmp", state_path)
    return model_path


def try_load_checkpoint(policy, optimizer, train_cfg: TrainConfig, exp_id: str):
    path = os.path.join(train_cfg.data_dir, exp_id)
    if not os.path.exists(path):
        print("No checkpoints found. Assuming new experiment")
        return

    trainer_path = os.path.join(path, "trainer_state.pt")
    resume_state = torch.load(trainer_path)
    model_path = os.path.join(path, resume_state["model_name"])
    policy.uncompiled.load_state_dict(model_path, map_location=train_cfg.device)
    optimizer.load_state_dict(resume_state["optimizer_state_dict"])
    print(f"Loaded checkpoint {resume_state['model_name']}")


def count_params(policy):
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


def seed_everything(seed, torch_deterministic):
    print(f"Seeding Everything with seed: {seed}" + ("and torch_deterministic" if torch_deterministic else ""))
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


ROUND_OPEN = rich.box.Box("╭──╮\n│  │\n│  │\n│  │\n│  │\n│  │\n│  │\n╰──╯\n")


c1 = "[bright_cyan]"
c2 = "[white]"
c3 = "[cyan]"
b1 = "[bright_cyan]"
b2 = "[bright_white]"


def abbreviate(num):
    if num < 1e3:
        return f"{b2}{num:.0f}"
    elif num < 1e6:
        return f"{b2}{num / 1e3:.1f}{c2}k"
    elif num < 1e9:
        return f"{b2}{num / 1e6:.1f}{c2}m"
    elif num < 1e12:
        return f"{b2}{num / 1e9:.1f}{c2}b"
    else:
        return f"{b2}{num / 1e12:.1f}{c2}t"


def duration(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, time, uptime):
    percent = 0 if uptime == 0 else int(100 * time / uptime - 1e-5)
    return f"{c1}{name}", duration(time), f"{b2}{percent:2d}%"


def print_dashboard(env_name, utilization, global_step, epoch, profile, losses, stats, msg, clear=False, max_stats=[0]):
    console = Console()
    if clear:
        console.clear()

    dashboard = Table(box=ROUND_OPEN, expand=True, show_header=False, border_style="bright_cyan")

    table = Table(box=None, expand=True, show_header=False)
    dashboard.add_row(table)
    cpu_percent = np.mean(utilization.cpu_util)
    dram_percent = np.mean(utilization.cpu_mem)
    gpu_percent = np.mean(utilization.gpu_util)
    vram_percent = np.mean(utilization.gpu_mem)
    table.add_column(justify="left", width=30)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=12)
    table.add_column(justify="center", width=13)
    table.add_column(justify="right", width=13)
    table.add_row(
        f":blowfish: {c1}PufferLib {b2}2.0.0",
        f"{c1}CPU: {c3}{cpu_percent:.1f}%",
        f"{c1}GPU: {c3}{gpu_percent:.1f}%",
        f"{c1}DRAM: {c3}{dram_percent:.1f}%",
        f"{c1}VRAM: {c3}{vram_percent:.1f}%",
    )

    s = Table(box=None, expand=True)
    s.add_column(f"{c1}Summary", justify="left", vertical="top", width=16)
    s.add_column(f"{c1}Value", justify="right", vertical="top", width=8)
    s.add_row(f"{c2}Environment", f"{b2}{env_name}")
    s.add_row(f"{c2}Agent Steps", abbreviate(global_step))
    s.add_row(f"{c2}SPS", abbreviate(profile.SPS))
    s.add_row(f"{c2}Epoch", abbreviate(epoch))
    s.add_row(f"{c2}Uptime", duration(profile.uptime))
    s.add_row(f"{c2}Remaining", duration(profile.remaining))

    p = Table(box=None, expand=True, show_header=False)
    p.add_column(f"{c1}Performance", justify="left", width=10)
    p.add_column(f"{c1}Time", justify="right", width=8)
    p.add_column(f"{c1}%", justify="right", width=4)
    p.add_row(*fmt_perf("Evaluate", profile.eval_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.eval_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Env", profile.env_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.eval_misc_time, profile.uptime))
    p.add_row(*fmt_perf("Train", profile.train_time, profile.uptime))
    p.add_row(*fmt_perf("  Forward", profile.train_forward_time, profile.uptime))
    p.add_row(*fmt_perf("  Learn", profile.learn_time, profile.uptime))
    p.add_row(*fmt_perf("  Misc", profile.train_misc_time, profile.uptime))

    ltb = Table(
        box=None,
        expand=True,
    )
    ltb.add_column(f"{c1}Losses", justify="left", width=16)
    ltb.add_column(f"{c1}Value", justify="right", width=8)
    for metric, value in asdict(losses).items():
        ltb.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

    monitor = Table(box=None, expand=True, pad_edge=False)
    monitor.add_row(s, p, ltb)
    dashboard.add_row(monitor)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    left = Table(box=None, expand=True)
    right = Table(box=None, expand=True)
    table.add_row(left, right)
    left.add_column(f"{c1}User Stats", justify="left", width=20)
    left.add_column(f"{c1}Value", justify="right", width=10)
    right.add_column(f"{c1}User Stats", justify="left", width=20)
    right.add_column(f"{c1}Value", justify="right", width=10)

    i = 0
    for metric, value in stats.items():
        try:  # Discard non-numeric values
            int(value)
        except:  # noqa
            continue

        u = left if i % 2 == 0 else right
        u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
        i += 1
        if i == 30:
            break

    for i in range(max_stats[0] - i):
        u = left if i % 2 == 0 else right
        u.add_row("", "")

    max_stats[0] = max(max_stats[0], i)

    table = Table(box=None, expand=True, pad_edge=False)
    dashboard.add_row(table)
    table.add_row(f" {c1}Message: {c2}{msg}")

    with console.capture() as capture:
        console.print(dashboard)

    print("\033[0;0H" + capture.get())
