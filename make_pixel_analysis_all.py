from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed.")
    print("Run: pip install matplotlib")
    raise

# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------

gt_dir = Path("/data/DATASETS/DIV2K/DIV2K_valid_HR")

a_dir = Path(
    "/data/ai23btech11029/SAFMN/results/"
    "SAFMN_c36n8_x4_archived_20260914_222244/"
    "visualization/DIV2K_val100"
)

b25_dir = Path(
    "/data/ai23btech11029/SAFMN/results/"
    "SAFMN_c36n8_x4/visualization/DIV2K_val100_B25"
)

b10_dir = Path(
    "/data/ai23btech11029/SAFMN/results/"
    "SAFMN_c36n8_x4/visualization/DIV2K_val100_B10"
)

out_dir = Path(
    "/data/ai23btech11029/SAFMN/"
    "visual_comparisons/pixel_analysis_all"
)

out_dir.mkdir(parents=True, exist_ok=True)

# Same five images used previously
names = ["0883", "0886", "0890", "0896", "0900"]

# ------------------------------------------------------------
# FUNCTION
# ------------------------------------------------------------

def load_image(path):
    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.int16
    )


def analyze(model_name, model_dir, filename_template):
    results = []

    for name in names:

        gt_path = gt_dir / f"{name}.png"
        model_path = model_dir / filename_template.format(name)

        if not gt_path.exists():
            print(f"[{model_name}] Missing GT: {gt_path}")
            continue

        if not model_path.exists():
            print(f"[{model_name}] Missing output: {model_path}")
            continue

        gt = load_image(gt_path)
        pred = load_image(model_path)

        if gt.shape != pred.shape:
            print(
                f"[{model_name}] Shape mismatch {name}: "
                f"GT={gt.shape}, prediction={pred.shape}"
            )
            continue

        # Absolute RGB difference
        diff = np.abs(gt - pred)

        # Average over RGB channels
        diff_gray = diff.mean(axis=2)

        mae = diff.mean()
        max_diff = int(diff.max())

        # Exact pixel = all R,G,B values exactly equal
        exact_pixels = int(np.all(diff == 0, axis=2).sum())

        total_pixels = gt.shape[0] * gt.shape[1]

        exact_percentage = (
            100.0 * exact_pixels / total_pixels
        )

        results.append(
            (
                name,
                mae,
                max_diff,
                exact_pixels,
                total_pixels,
                exact_percentage,
            )
        )

        # ----------------------------------------------------
        # 1. AUTO-CONTRAST GRAYSCALE MAP
        # ----------------------------------------------------
        # Use percentile instead of 0-255 so small differences
        # become visible.

        p99 = np.percentile(diff_gray, 99)

        if p99 <= 0:
            p99 = 1

        gray_visual = np.clip(
            diff_gray / p99 * 255,
            0,
            255
        ).astype(np.uint8)

        gray_img = Image.fromarray(
            gray_visual,
            mode="L"
        )

        gray_path = out_dir / f"{name}_{model_name}_difference_gray.png"
        gray_img.save(gray_path)

        # ----------------------------------------------------
        # 2. COLOR HEATMAP
        # ----------------------------------------------------

        plt.figure(figsize=(8, 6))

        plt.imshow(
            diff_gray,
            cmap="turbo",
            vmin=0,
            vmax=p99
        )

        plt.colorbar(
            label="Absolute pixel difference"
        )

        plt.title(
            f"{model_name} vs Ground Truth - {name}"
        )

        plt.axis("off")

        heatmap_path = (
            out_dir /
            f"{name}_{model_name}_difference_heatmap.png"
        )

        plt.tight_layout()
        plt.savefig(
            heatmap_path,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

    return results


# ------------------------------------------------------------
# RUN ALL THREE MODELS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("PIXEL-LEVEL ANALYSIS")
print("=" * 90)

a_results = analyze(
    "ModelA",
    a_dir,
    "{}x4_SAFMN_c36n8_x4.png"
)

b25_results = analyze(
    "ModelB25",
    b25_dir,
    "{}x4_SAFMN_c36n8_x4.png"
)

b10_results = analyze(
    "ModelB10",
    b10_dir,
    "{}x4_SAFMN_c36n8_x4.png"
)


# ------------------------------------------------------------
# PRINT TABLE
# ------------------------------------------------------------

def print_results(title, results):

    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)

    print(
        f"{'Image':<10}"
        f"{'MAE':>12}"
        f"{'Max':>10}"
        f"{'Exact Pixels':>18}"
        f"{'Exact %':>12}"
    )

    for r in results:
        name, mae, max_diff, exact, total, pct = r

        print(
            f"{name:<10}"
            f"{mae:>12.4f}"
            f"{max_diff:>10}"
            f"{exact:>18}"
            f"{pct:>11.4f}%"
        )


print_results("MODEL A", a_results)
print_results("MODEL B-25", b25_results)
print_results("MODEL B-10", b10_results)


# ------------------------------------------------------------
# CREATE CSV-LIKE TEXT FILE
# ------------------------------------------------------------

summary_path = out_dir / "pixel_results.txt"

with open(summary_path, "w") as f:

    f.write("Pixel-level comparison against Ground Truth\n")
    f.write("=" * 90 + "\n\n")

    for model_name, results in [
        ("Model A", a_results),
        ("Model B-25", b25_results),
        ("Model B-10", b10_results),
    ]:

        f.write(model_name + "\n")
        f.write("-" * 90 + "\n")

        f.write(
            "Image,MAE,MaxDifference,ExactPixels,"
            "TotalPixels,ExactPercentage\n"
        )

        for r in results:
            name, mae, max_diff, exact, total, pct = r

            f.write(
                f"{name},{mae:.4f},{max_diff},"
                f"{exact},{total},{pct:.4f}\n"
            )

        f.write("\n")


# ------------------------------------------------------------
# CREATE SIDE-BY-SIDE HEATMAP COMPARISON
# ------------------------------------------------------------

print("\nCreating combined heatmaps...")

for name in names:

    gt_path = gt_dir / f"{name}.png"

    a_path = a_dir / f"{name}x4_SAFMN_c36n8_x4.png"
    b25_path = b25_dir / f"{name}x4_SAFMN_c36n8_x4.png"
    b10_path = b10_dir / f"{name}x4_SAFMN_c36n8_x4.png"

    if not (
        gt_path.exists()
        and a_path.exists()
        and b25_path.exists()
        and b10_path.exists()
    ):
        print(f"Skipping combined heatmap for {name}")
        continue

    gt = load_image(gt_path)
    a = load_image(a_path)
    b25 = load_image(b25_path)
    b10 = load_image(b10_path)

    diff_a = np.abs(gt - a).mean(axis=2)
    diff_b25 = np.abs(gt - b25).mean(axis=2)
    diff_b10 = np.abs(gt - b10).mean(axis=2)

    # Same scale for all three models.
    combined_max = np.percentile(
        np.concatenate([
            diff_a.flatten(),
            diff_b25.flatten(),
            diff_b10.flatten()
        ]),
        99
    )

    if combined_max <= 0:
        combined_max = 1

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    data = [
        ("Model A", diff_a),
        ("Model B-25", diff_b25),
        ("Model B-10", diff_b10),
    ]

    for ax, (title, diff) in zip(axes, data):

        im = ax.imshow(
            diff,
            cmap="turbo",
            vmin=0,
            vmax=combined_max
        )

        ax.set_title(
            f"{title} vs Ground Truth\nImage {name}"
        )

        ax.axis("off")

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="Absolute pixel difference"
    )

    fig.suptitle(
        f"Pixel Difference Comparison - {name}"
    )

    combined_path = (
        out_dir /
        f"{name}_A_B25_B10_heatmap_comparison.png"
    )

    plt.tight_layout()
    plt.savefig(
        combined_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

print(f"\nAll results saved to:")
print(out_dir)

print(f"\nNumerical results:")
print(summary_path)

print("\nGenerated:")
print("- Auto-contrast grayscale difference maps")
print("- Color heatmaps")
print("- Combined A vs B-25 vs B-10 heatmaps")
print("- Pixel-level numerical results")
