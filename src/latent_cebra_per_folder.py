"""
Per-sequence (object + folder) pipeline:
  HO3D rows -> CVAE posterior mean latent (ordered by frame) -> CEBRA with time -> PNG.

Output layout: <project>/output/<object_name>/<folder_name>.png
Example: output/003_cracker_box/MC1.png
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path


def _log(msg: str) -> None:
    """Always flush so PyCharm / IDEs show output while imports or long fits run."""
    print(msg, flush=True)


_log("latent_cebra_per_folder: starting (PyTorch + CEBRA imports can take 30–90s — wait for the next lines).")

_SRC = Path(__file__).resolve().parent
_SRC_RESOLVED = _SRC.resolve()

# Python prepends the script directory to sys.path, so a bare `import cebra` loads
# `src/cebra.py` instead of the pip package. Temporarily drop this dir, import the
# library, then restore sys.path (same pattern as avoiding name shadowing).
_log("  importing pip package 'cebra'…")
_sys_path_saved = list(sys.path)
sys.path[:] = [p for p in sys.path if not (p and Path(p).resolve() == _SRC_RESOLVED)]
import cebra  # pip package
sys.path[:] = _sys_path_saved
_log("  cebra (pip) import done.")

_log("  importing NumPy, Matplotlib (non-GUI backend), PyTorch…")
import matplotlib

matplotlib.use("Agg")  # avoid Tk/Qt startup delays or hidden GUI when only saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_log(f"  PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cvae_model import CVAE

_log("  CVAE module loaded. Beginning main work.\n")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_split_arrays(
    dataset_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    split = split.lower()
    if split == "train":
        hand = np.asarray(data["hand_train"], dtype=np.float64)
        obj = np.asarray(data["obj_train"], dtype=np.float64)
        idx = np.asarray(data["train_indices"], dtype=np.int64)
    elif split == "val":
        hand = np.asarray(data["hand_val"], dtype=np.float64)
        obj = np.asarray(data["obj_val"], dtype=np.float64)
        idx = np.asarray(data["val_indices"], dtype=np.int64)
    elif split == "test":
        hand = np.asarray(data["hand_test"], dtype=np.float64)
        obj = np.asarray(data["obj_test"], dtype=np.float64)
        idx = np.asarray(data["test_indices"], dtype=np.int64)
    else:
        raise ValueError("split must be one of: train, val, test")

    obj_names = np.asarray(data["obj_names"], dtype=str)[idx]
    folder_names = np.asarray(data["folder_names"], dtype=str)[idx]
    frame_numbers = np.asarray(data["frame_numbers"], dtype=str)[idx]
    return hand, obj, obj_names, folder_names, frame_numbers


def compute_posterior_mean_latent(
    model: CVAE,
    obj: np.ndarray,
    hand: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Posterior mean mu_q for each row (same convention as cebra.compute_latent)."""
    model.eval()
    n = obj.shape[0]
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ob = torch.as_tensor(obj[start:end], dtype=torch.float32, device=device)
            ha = torch.as_tensor(hand[start:end], dtype=torch.float32, device=device)
            c = model.object_encoder(ob)
            h = model.hand_encoder(ha)
            mu_q, _ = model.posterior_net(h, c)
            out.append(mu_q.cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_cebra_time(
    latent: np.ndarray,
    *,
    output_dim: int,
    max_iterations: int,
    batch_size: int,
    device: str,
    verbose: bool,
) -> np.ndarray:
    """CEBRA embedding using within-sequence time indices 0..T-1."""
    t = np.arange(latent.shape[0], dtype=np.float32).reshape(-1, 1)
    bs = min(batch_size, max(latent.shape[0], 1))
    model = cebra.CEBRA(
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=max_iterations,
        batch_size=bs,
        learning_rate=3e-4,
        temperature=1.0,
        output_dimension=output_dim,
        distance="cosine",
        device=device,
        verbose=verbose,
    )
    model.fit(latent.astype(np.float32), t)
    return model.transform(latent.astype(np.float32))


def plot_cebra_time_2d(
    emb: np.ndarray,
    time_1d: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """2D scatter (dims 0–1) colored by time; matches the style of cebra.plot_2Dcebra_time."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=time_1d,
        cmap="viridis",
        alpha=0.75,
        s=18,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time index (sorted frames)", rotation=270, labelpad=18)
    ax.set_xlabel("CEBRA 1")
    ax.set_ylabel("CEBRA 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CEBRA (time) on CVAE latent per object/folder sequence.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_project_root() / "dataset" / "hand_object_data.pkl",
        help="Path to hand_object_data.pkl",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=_SRC / "cvae_weight.pth",
        help="Path to CVAE state dict",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root() / "output",
        help="Root folder for output/<object>/<folder>.png",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--latent-dim", type=int, default=32, help="Must match checkpoint (see src/cebra.py).")
    parser.add_argument("--cebra-dim", type=int, default=3, help="CEBRA output dimension (plot uses first 2).")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--cebra-batch-size", type=int, default=512)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--verbose-cebra", action="store_true")
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm bar (still prints _log lines).",
    )
    args = parser.parse_args()

    device_torch = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device_cebra = "cuda" if device_torch.type == "cuda" else "cpu"

    _log(f"Dataset: {args.dataset.resolve()}")
    _log(f"Split: {args.split} | CVAE weights: {args.weights.resolve()} | device: {device_torch}")
    _log(f"CEBRA: max_iterations={args.max_iterations}, output_dim={args.cebra_dim} (no console spam unless --verbose-cebra)")

    _log("Loading pickle (large file may take a bit)…")
    hand, obj, obj_names, folder_names, frame_numbers = load_split_arrays(args.dataset, args.split)
    _log(f"Loaded arrays: hand {hand.shape}, obj {obj.shape}")

    _log("Loading CVAE state dict…")
    model = CVAE(latent_dim=args.latent_dim)
    state = torch.load(args.weights, map_location=device_torch)
    model.load_state_dict(state)
    model.to(device_torch)

    # Group row indices by (object, folder)
    key_to_rows: dict[tuple[str, str], list[int]] = {}
    for row in range(hand.shape[0]):
        key = (str(obj_names[row]), str(folder_names[row]))
        key_to_rows.setdefault(key, []).append(row)

    total_groups = len(key_to_rows)
    _log(f"Found {total_groups} (object, folder) sequences. Each runs CEBRA ({args.max_iterations} iters) — this takes a long time.\n")

    iterator = sorted(key_to_rows.items())
    if args.no_progress_bar:
        prog = iterator
    else:
        prog = tqdm(
            iterator,
            desc="CEBRA per sequence",
            unit="seq",
            file=sys.stderr,
            mininterval=0.5,
            dynamic_ncols=True,
        )

    for (obj_name, folder_name), rows in prog:
        if isinstance(prog, tqdm):
            prog.set_postfix_str(f"{obj_name}/{folder_name}", refresh=False)
        rows_arr = np.asarray(rows, dtype=np.int64)
        fr_int = frame_numbers[rows_arr].astype(np.int64)
        order = np.argsort(fr_int, kind="mergesort")
        idx_sorted = rows_arr[order]

        obj_seq = obj[idx_sorted]
        hand_seq = hand[idx_sorted]
        _log(f"  → {obj_name} / {folder_name}: encoding {hand_seq.shape[0]} frames…")
        latent = compute_posterior_mean_latent(
            model,
            obj_seq,
            hand_seq,
            device_torch,
            batch_size=args.encode_batch_size,
        )
        time_idx = np.arange(latent.shape[0], dtype=np.float32)

        _log(f"     CEBRA fit (silent unless --verbose-cebra)…")
        emb = fit_cebra_time(
            latent,
            output_dim=args.cebra_dim,
            max_iterations=args.max_iterations,
            batch_size=args.cebra_batch_size,
            device=device_cebra,
            verbose=args.verbose_cebra,
        )

        out_dir = args.output_dir / obj_name
        out_path = out_dir / f"{folder_name}.png"
        title = f"CEBRA (time) on CVAE latent\n{args.split} | {obj_name} | {folder_name} | T={latent.shape[0]}"
        plot_cebra_time_2d(emb, time_idx, title, out_path)

        _log(f"     saved {out_path.resolve()}")

    _log(f"Finished. Plots under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    # Line-buffer stdout/stderr when supported (helps PyCharm show logs immediately).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError):
            pass
    # Allow running from any cwd without changing HO3D paths (none required here).
    os.chdir(_project_root())
    main()
