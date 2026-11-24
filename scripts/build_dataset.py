"""Extract `(4, 64, 800)` log-mel caches and build multimodal dataset manifests for MUSDB18-HQ."""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from wadidaw.data.dataset import (
    CACHE_ROOT,
    SAMPLES_ROOT,
    build_human_test_samples,
    build_split_samples,
    extract_and_cache_song_excerpts,
)
from wadidaw.data.musdb import (
    MUSDB_ROOT,
    SPLIT_MANIFEST_PATH,
    create_musdb_split_manifest,
    load_split_manifest,
)
from wadidaw.data.prompts import save_prompt_banks


# Deteksi sekali apakah extractor menerima kwargs GPU-related.
_EXTRACT_PARAMS = set(inspect.signature(extract_and_cache_song_excerpts).parameters)
_SUPPORTS_DEVICE_KW = {"device", "batch_size", "use_amp"}.issubset(_EXTRACT_PARAMS)


def _configure_torch(device: str) -> None:
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] {name} diminta tapi CUDA tidak tersedia -> fallback ke CPU")
        return "cpu"
    return name


def _process_one_song(
    task: tuple[
        dict[str, object], str, int, Path, Path, int, str, bool, str, int, bool
    ],
) -> list[dict[str, object]]:
    (
        member,
        split,
        song_number,
        musdb_root,
        cache_root,
        n_excerpts,
        dtype,
        overwrite,
        device,
        batch_size,
        use_amp,
    ) = task

    # Round-robin GPU per worker kalau >1 GPU.
    if device.startswith("cuda") and torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            device = f"cuda:{song_number % n_gpu}"
        _configure_torch(device)

    kwargs = dict(
        song_member=member,
        split=split,
        song_number=song_number,
        musdb_root=musdb_root,
        cache_root=cache_root,
        n_excerpts=n_excerpts,
        dtype=dtype,
        overwrite=overwrite,
    )
    if _SUPPORTS_DEVICE_KW:
        kwargs.update(device=device, batch_size=batch_size, use_amp=use_amp)
    return extract_and_cache_song_excerpts(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MUSDB18-HQ log-mel cache and multimodal (audio, prompt, gain) manifests."
    )
    parser.add_argument("--musdb-root", type=Path, default=MUSDB_ROOT)
    parser.add_argument("--manifest", type=Path, default=SPLIT_MANIFEST_PATH)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--excerpts-per-song", type=int, default=10)
    parser.add_argument("--prompts-per-excerpt", type=int, default=4)
    parser.add_argument(
        "--max-songs",
        type=int,
        default=None,
        help="Optional cap per split for quick runs.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--overwrite", action="store_true")
    # ---- CUDA / throughput flags ----
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' | 'cpu' | 'cuda' | 'cuda:N' (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size untuk GPU mel extraction (diabaikan di CPU).",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable torch.cuda.amp autocast (CUDA only)."
    )
    parser.add_argument(
        "--start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help="Multiprocessing start method; 'spawn' aman untuk CUDA.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    use_amp = bool(args.amp and device.startswith("cuda"))
    _configure_torch(device)

    if device.startswith("cuda"):
        n_gpu = torch.cuda.device_count()
        total_vram = (
            sum(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu))
            / 1e9
        )
        print(
            f"[build] device={device} | gpus={n_gpu} | total_vram={total_vram:.1f} GB | "
            f"amp={use_amp} | batch={args.batch_size} | workers={args.workers}"
        )
        if args.workers > 1:
            print(
                f"[build] perhatian: {args.workers} worker → {args.workers} CUDA context. "
                f"Turunkan --workers kalau VRAM OOM."
            )
        if not _SUPPORTS_DEVICE_KW:
            print(
                "[warn] extract_and_cache_song_excerpts() tidak menerima "
                "device/batch_size/use_amp; flag GPU akan diabaikan di worker. "
                "Lihat patch #2 di bawah."
            )
    else:
        print(
            f"[build] device=cpu | workers={args.workers} (tidak ada GPU acceleration)"
        )

    if not args.manifest.exists():
        create_musdb_split_manifest(
            musdb_root=args.musdb_root,
            output_path=args.manifest,
            seed=args.seed,
        )

    manifest = load_split_manifest(args.manifest)
    save_prompt_banks(args.samples_root)

    cached_by_split: dict[str, list[dict[str, object]]] = {}

    # spawn: CUDA-safe di Linux (fork + CUDA sering hang / error "Cannot re-initialize CUDA").
    mp_ctx = mp.get_context(args.start_method)

    for split in ("train", "validation", "test"):
        members = list(manifest["splits"][split])
        if args.max_songs is not None:
            members = members[: args.max_songs]

        tasks = [
            (
                member,
                split,
                idx,
                args.musdb_root,
                args.cache_root,
                args.excerpts_per_song,
                args.dtype,
                args.overwrite,
                device,
                args.batch_size,
                use_amp,
            )
            for idx, member in enumerate(members, start=1)
        ]

        print(
            f"Extracting {split} log-mel cache "
            f"({len(tasks)} songs × {args.excerpts_per_song} excerpts) on {device}..."
        )
        excerpts_list: list[dict[str, object]] = []
        if args.workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp_ctx
            ) as pool:
                for song_excerpts in pool.map(_process_one_song, tasks):
                    excerpts_list.extend(song_excerpts)
        else:
            for t in tasks:
                excerpts_list.extend(_process_one_song(t))

        cached_by_split[split] = excerpts_list
        active_4_count = sum(1 for ex in excerpts_list if ex["all_stems_active"])
        print(
            f"  -> {len(excerpts_list)} excerpts cached "
            f"({active_4_count}/{len(excerpts_list)} have all 4 stems > -45 dBFS)"
        )

    args.samples_root.mkdir(parents=True, exist_ok=True)
    seed_offsets = {"train": 100, "validation": 200, "test": 300}

    for split in ("train", "validation", "test"):
        samples = build_split_samples(
            cached_excerpts=cached_by_split[split],
            split=split,
            n_prompts_per_excerpt=args.prompts_per_excerpt,
            seed=args.seed + seed_offsets[split],
            perturb_g0=True,
        )
        out_name = (
            f"{split}_samples.json"
            if split != "test"
            else "test_synthetic_samples.json"
        )
        out_path = args.samples_root / out_name
        out_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved {len(samples):4d} multimodal samples -> {out_path}")

    human_test_samples = build_human_test_samples(
        cached_test_excerpts=cached_by_split["test"],
        seed=args.seed,
        perturb_g0=True,
    )
    human_out_path = args.samples_root / "test_human_samples.json"
    human_out_path.write_text(
        json.dumps(human_test_samples, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Saved {len(human_test_samples):4d} human test samples -> {human_out_path}")


if __name__ == "__main__":
    main()
