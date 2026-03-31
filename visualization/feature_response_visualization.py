import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime


def save_activation_map(
    feat,
    orig_img,
    layer_name,
    save_dir="./visualization/activation_layers",
    flip_vertical=True,
):
    """
    Save visualization-oriented feature response maps.
    Reference: HCPMNet visualization style.
    Args:
        feat: [1, C, H, W] tensor.
        orig_img: [1, 3, H, W] tensor.
        layer_name: layer name string.
        save_dir: output directory.
        flip_vertical: whether to flip vertically before saving.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Resize feature map to image size
    feat_map = F.interpolate(
        feat, size=orig_img.shape[-2:], mode="bilinear", align_corners=False
    )

    # feat_map = feat_map.mean(dim=1, keepdim=False).squeeze(0).detach().cpu().numpy()

    # Normalize to 0-255
    feat_map = feat_map - feat_map.min()
    feat_map = feat_map / (feat_map.max() + 1e-5)
    feat_map = (feat_map * 255).astype(np.uint8)

    # Generate heatmap
    heatmap = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)

    # Prepare original image
    img_np = orig_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    img_np = img_np - img_np.min()
    img_np = img_np / (img_np.max() + 1e-5)
    img_np = (img_np * 255).astype(np.uint8)

    # Convert image to BGR for OpenCV blending and saving
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)


    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)


    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    pid = os.getpid()
    uniq_id = f"{ts}_{pid}"

    if flip_vertical:
        overlay = np.flip(overlay, axis=0)
        img_bgr = np.flip(img_bgr, axis=0)


    save_path = os.path.join(save_dir, f"{layer_name}_overlay_{uniq_id}.png")
    cv2.imwrite(save_path, overlay)
    print(f"[Saved] {save_path}")

    # Save original image
    orig_path = os.path.join(save_dir, f"{layer_name}_orig_{uniq_id}.png")
    cv2.imwrite(orig_path, img_bgr)
    print(f"[Saved] {orig_path}")