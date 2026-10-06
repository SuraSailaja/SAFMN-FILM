from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
    "SAFMN_c36n8_x4_archived_20260915_121835/"
    "visualization/DIV2K_val100_B25"
)

b10_dir = Path(
    "/data/ai23btech11029/SAFMN/results/"
    "SAFMN_c36n8_x4/"
    "visualization/DIV2K_val100_B10"
)

out_dir = Path(
    "/data/ai23btech11029/SAFMN/"
    "visual_comparisons/three_model_heatmaps"
)

out_dir.mkdir(parents=True, exist_ok=True)

names = ["0883", "0886", "0890", "0896", "0900"]


def load(path):
    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.int16
    )


print("=" * 90)
print("THREE-MODEL PIXEL DIFFERENCE HEATMAPS")
print("=" * 90)

for name in names:

    gt_path = gt_dir / f"{name}.png"

    a_path = a_dir / f"{name}x4_SAFMN_c36n8_x4.png"
    b25_path = b25_dir / f"{name}x4_SAFMN_c36n8_x4.png"
    b10_path = b10_dir / f"{name}x4_SAFMN_c36n8_x4.png"

    paths = [gt_path, a_path, b25_path, b10_path]

    missing = [str(p) for p in paths if not p.exists()]

    if missing:
        print(f"\nSkipping {name} because files are missing:")
        for p in missing:
            print(" ", p)
        continue

    gt = load(gt_path)
    a = load(a_path)
    b25 = load(b25_path)
    b10 = load(b10_path)

    # --------------------------------------------------------
    # RGB absolute differences
    # --------------------------------------------------------

    diff_a = np.abs(gt - a).mean(axis=2)
    diff_b25 = np.abs(gt - b25).mean(axis=2)
    diff_b10 = np.abs(gt - b10).mean(axis=2)

    # --------------------------------------------------------
    # Use ONE common scale for all three models.
    #
    # 99th percentile avoids a few extreme pixels dominating
    # the visualization.
    # --------------------------------------------------------

    all_values = np.concatenate([
        diff_a.ravel(),
        diff_b25.ravel(),
        diff_b10.ravel()
    ])

    vmax = np.percentile(all_values, 99)

    if vmax <= 0:
        vmax = 1

    # --------------------------------------------------------
    # Three-way heatmap
    # --------------------------------------------------------

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
            vmax=vmax
        )

        ax.set_title(
            f"{title}\nImage {name}"
        )

        ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )

    cbar.set_label(
        "Mean absolute RGB pixel difference"
    )

    fig.suptitle(
        f"Pixel Difference from Ground Truth — {name}",
        fontsize=16
    )

    plt.tight_layout()

    output = out_dir / f"{name}_three_model_heatmap.png"

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nCreated: {output}")
    print(f"Common heatmap maximum: {vmax:.4f}")


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

print(f"\nHeatmaps are here:")
print(out_dir)
