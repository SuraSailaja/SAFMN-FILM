from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

hr_dir = Path("/data/DATASETS/DIV2K/DIV2K_valid_HR")

a_dir = Path("/data/ai23btech11029/SAFMN/results/SAFMN_c36n8_x4_archived_20260914_222244/visualization/DIV2K_val100")
b25_dir = Path("/data/ai23btech11029/SAFMN/results/SAFMN_c36n8_x4/visualization/DIV2K_val100_B25")

out_dir = Path("/data/ai23btech11029/SAFMN/visual_comparisons/difference_maps")
out_dir.mkdir(parents=True, exist_ok=True)

names = ["0883", "0886", "0890", "0896", "0900"]

for name in names:
    gt = np.array(
        Image.open(hr_dir / f"{name}.png").convert("RGB"),
        dtype=np.int16
    )

    model_a = np.array(
        Image.open(a_dir / f"{name}x4_SAFMN_c36n8_x4.png").convert("RGB"),
        dtype=np.int16
    )

    model_b25 = np.array(
        Image.open(b25_dir / f"{name}x4_SAFMN_c36n8_x4.png").convert("RGB"),
        dtype=np.int16
    )

    # Exact absolute pixel difference for each RGB channel.
    diff_a = np.abs(model_a - gt)
    diff_b25 = np.abs(model_b25 - gt)

    # Mean absolute difference across RGB channels.
    diff_a_gray = np.mean(diff_a, axis=2).astype(np.uint8)
    diff_b25_gray = np.mean(diff_b25, axis=2).astype(np.uint8)

    # Save exact RGB difference maps.
    Image.fromarray(diff_a.astype(np.uint8)).save(
        out_dir / f"{name}_ModelA_RGB_difference.png"
    )

    Image.fromarray(diff_b25.astype(np.uint8)).save(
        out_dir / f"{name}_ModelB25_RGB_difference.png"
    )

    # Save grayscale mean absolute difference maps.
    Image.fromarray(diff_a_gray, mode="L").save(
        out_dir / f"{name}_ModelA_mean_difference.png"
    )

    Image.fromarray(diff_b25_gray, mode="L").save(
        out_dir / f"{name}_ModelB25_mean_difference.png"
    )

    # Print numerical pixel-error statistics.
    print(f"\n{name}")
    print(
        f"Model A   : MAE={diff_a.mean():.4f}, "
        f"max={diff_a.max()}, "
        f"pixels_exact={np.all(diff_a == 0, axis=2).sum()}"
    )
    print(
        f"Model B-25 : MAE={diff_b25.mean():.4f}, "
        f"max={diff_b25.max()}, "
        f"pixels_exact={np.all(diff_b25 == 0, axis=2).sum()}"
    )

print("\nDifference maps saved to:")
print(out_dir)
