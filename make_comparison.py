import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from basicsr.archs.safmn_arch import SAFMN


ROOT = "/data/ai23btech11029/SAFMN"

LR_DIR = "/data/DATASETS/DIV2K/DIV2K_valid_LR_bicubic/X4"
GT_DIR = "/data/DATASETS/DIV2K/DIV2K_valid_HR"

CHECKPOINTS = {
    "A": f"{ROOT}/experiments/SAFMN_ModelA_b256_lr0.002_52K_DF2K_x4/models/net_g_latest.pth",
    "B25": f"{ROOT}/experiments/SAFMN_ModelB25_b256_lr0.002_13K_DF2K_x4/models/net_g_latest.pth",
    "B10": f"{ROOT}/experiments/SAFMN_ModelB10_b256_lr0.002_5.2K_DF2K_x4/models/net_g_latest.pth",
    "C25": f"{ROOT}/experiments/SAFMN_ModelC25_b256_lr0.002_13K_DF2K_x4/models/net_g_latest.pth",
    "C10": f"{ROOT}/experiments/SAFMN_ModelC10_b256_lr0.002_5.2K_DF2K_x4/models/net_g_latest.pth",
}

OUT_DIR = f"{ROOT}/visual_comparisons/ABC_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# Same five clean DIV2K validation images for every model
SAMPLES = ["0883", "0886", "0890", "0896", "0900"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


def load_model(checkpoint):

    model = SAFMN(
        dim=36,
        n_blocks=8,
        ffn_scale=2.0,
        upscaling_factor=4
    )

    ckpt = torch.load(checkpoint, map_location=DEVICE)

    if "params_ema" in ckpt:
        state_dict = ckpt["params_ema"]
    elif "params" in ckpt:
        state_dict = ckpt["params"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(DEVICE)

    return model


def read_rgb(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def lr_to_tensor(img):

    tensor = torch.from_numpy(
        img.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return tensor


def tensor_to_image(tensor):

    img = tensor.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0).numpy()

    return np.clip(img, 0, 1)


# ============================================================
# Load all five models
# ============================================================

models = {}

for name, checkpoint in CHECKPOINTS.items():

    print(f"\nLoading {name}")
    print(checkpoint)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    models[name] = load_model(checkpoint)

print("\nAll five models loaded successfully.")


# ============================================================
# Generate comparisons
# ============================================================

for sample in SAMPLES:

    print("\n========================================")
    print("Processing:", sample)
    print("========================================")

    lr_path = os.path.join(
        LR_DIR,
        f"{sample}x4.png"
    )

    gt_path = os.path.join(
        GT_DIR,
        f"{sample}.png"
    )

    lr = read_rgb(lr_path)
    gt = read_rgb(gt_path)

    # --------------------------------------------------------
    # Bicubic-upscaled LR ONLY for display
    # --------------------------------------------------------

    h, w = gt.shape[:2]

    lr_display = cv2.resize(
        lr,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    lr_display = lr_display.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # SAME LR input for every model
    # --------------------------------------------------------

    lq = lr_to_tensor(lr).to(DEVICE)

    outputs = {}

    with torch.no_grad():

        for name in ["A", "B25", "B10", "C25", "C10"]:

            print(f"Running {name}...")

            pred = models[name](lq)

            outputs[name] = tensor_to_image(pred)

    gt_display = gt.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Seven panels
    # --------------------------------------------------------

    images = [
        lr_display,
        outputs["A"],
        outputs["B25"],
        outputs["B10"],
        outputs["C25"],
        outputs["C10"],
        gt_display,
    ]

    titles = [
        "Input LR\n(Bicubic display)",
        "Model A\n100% data",
        "Model B25\n25% data",
        "Model B10\n10% data",
        "Model C25\n25% + Heavy Aug.",
        "Model C10\n10% + Heavy Aug.",
        "Ground Truth",
    ]

    fig, axes = plt.subplots(
        1,
        7,
        figsize=(28, 5)
    )

    for ax, image, title in zip(
        axes,
        images,
        titles
    ):

        ax.imshow(image)
        ax.set_title(
            title,
            fontsize=12
        )
        ax.axis("off")

    fig.suptitle(
        f"SAFMN Visual Comparison - DIV2K {sample}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUT_DIR,
        f"{sample}_ABC_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


print("\n========================================")
print("DONE")
print("========================================")
print("Output:")
print(OUT_DIR)
