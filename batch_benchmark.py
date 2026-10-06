import os
import gc
import time
import math
import torch
import torch.nn as nn

from basicsr.archs.safmn_arch import SAFMN

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = torch.device("cuda")
BATCH_SIZES = [64, 128, 256, 512, 1024]

WARMUP_ITERS = 10
MEASURE_ITERS = 30

HR_SIZE = 256
LR_SIZE = HR_SIZE // 4

BASE_BATCH = 64
BASE_LR = 1e-3

# ------------------------------------------------------------
# Check GPU
# ------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print()

results = []

# ------------------------------------------------------------
# Benchmark each batch size
# ------------------------------------------------------------
for batch_size in BATCH_SIZES:

    print("=" * 65)
    print(f"Testing batch size: {batch_size}")
    print("=" * 65)

    model = None
    optimizer = None
    lr_input = None
    target = None

    try:
        # Clean up previous test
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Exact SAFMN architecture from this repository
        model = SAFMN(
            dim=36,
            n_blocks=8,
            ffn_scale=2.0,
            upscaling_factor=4
        ).to(DEVICE)

        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=BASE_LR,
            betas=(0.9, 0.99)
        )

        # Real training-sized tensors:
        # LR = 64x64 -> HR = 256x256
        lr_input = torch.randn(
            batch_size, 3, LR_SIZE, LR_SIZE,
            device=DEVICE
        )
        target = torch.randn(
            batch_size, 3, HR_SIZE, HR_SIZE,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------
        for _ in range(WARMUP_ITERS):
            optimizer.zero_grad(set_to_none=True)

            output = model(lr_input)
            loss = torch.mean(torch.abs(output - target))

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # ----------------------------------------------------
        # Timed iterations
        # ----------------------------------------------------
        start = time.perf_counter()

        for _ in range(MEASURE_ITERS):
            optimizer.zero_grad(set_to_none=True)

            output = model(lr_input)
            loss = torch.mean(torch.abs(output - target))

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        sec_per_iter = elapsed / MEASURE_ITERS
        iter_per_sec = 1.0 / sec_per_iter

        eight_hour_iters = int((8 * 60 * 60) / sec_per_iter)

        multiplier = batch_size / BASE_BATCH
        scaled_lr = BASE_LR * math.sqrt(multiplier)

        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"Time for {MEASURE_ITERS} measured iterations: {elapsed:.2f} s")
        print(f"Seconds / iteration: {sec_per_iter:.4f}")
        print(f"Iterations / second: {iter_per_sec:.4f}")
        print(f"Estimated iterations in 8 hours: {eight_hour_iters:,}")
        print(f"Batch multiplier: {multiplier:.2f}x")
        print(f"Professor-scaled LR: {scaled_lr:.7f}")
        print(f"Peak allocated GPU memory: {peak_memory_gb:.2f} GB")
        print("STATUS: FITS")
        print()

        results.append({
            "batch": batch_size,
            "sec_per_iter": sec_per_iter,
            "iter_per_sec": iter_per_sec,
            "eight_hour_iters": eight_hour_iters,
            "multiplier": multiplier,
            "scaled_lr": scaled_lr,
            "peak_memory_gb": peak_memory_gb,
            "fits": True
        })

    except torch.cuda.OutOfMemoryError:
        print("STATUS: CUDA OUT OF MEMORY")
        print("Stopping batch-size sweep.")
        print()

        results.append({
            "batch": batch_size,
            "fits": False
        })

        break

    finally:
        del model
        del optimizer
        del lr_input
        del target

        gc.collect()
        torch.cuda.empty_cache()

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------
print()
print("=" * 90)
print("FINAL BATCH-SIZE SUMMARY")
print("=" * 90)

print(
    f"{'Batch':>8} "
    f"{'sec/iter':>12} "
    f"{'iter/sec':>12} "
    f"{'8h iters':>14} "
    f"{'LR':>12} "
    f"{'Peak GB':>10} "
    f"{'Status':>10}"
)

for r in results:
    if r["fits"]:
        print(
            f"{r['batch']:>8} "
            f"{r['sec_per_iter']:>12.4f} "
            f"{r['iter_per_sec']:>12.4f} "
            f"{r['eight_hour_iters']:>14,} "
            f"{r['scaled_lr']:>12.7f} "
            f"{r['peak_memory_gb']:>10.2f} "
            f"{'FIT':>10}"
        )
    else:
        print(
            f"{r['batch']:>8} "
            f"{'-':>12} "
            f"{'-':>12} "
            f"{'-':>14} "
            f"{'-':>12} "
            f"{'-':>10} "
            f"{'OOM':>10}"
        )

print("=" * 90)
