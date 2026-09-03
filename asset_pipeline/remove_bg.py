#!/usr/bin/env python3
"""Remove the background from a generated game asset (any body part).

Two stages:
  1. rembg alpha matte (neural) to find the rough subject.
  2. Color-key cleanup: pixels close to the (near-white) background color are
     turned transparent, so a faint gray ring rembg left behind is removed.

Works for any body-part render. Transparent background is preserved.

Usage:
  python remove_bg.py <input.png> [output.png] [--bg-dist 0.12] [--min-area 200]
"""
import argparse
import numpy as np
from PIL import Image
from rembg import remove
from skimage import measure

DEFAULT_SRC = "output/flux_head_canny_s0.2_00001_.png"
DEFAULT_DST = "transparent_asset.png"


def color_key(rgba, bg_dist=0.12, min_area=200):
    """Remove near-background colored pixels and keep the head silhouette."""
    arr = np.asarray(rgba.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[..., :3]

    # Estimate the background color from the image corners.
    corners = np.concatenate([
        rgb[:6, :6].reshape(-1, 3), rgb[:6, -6:].reshape(-1, 3),
        rgb[-6:, :6].reshape(-1, 3), rgb[-6:, -6:].reshape(-1, 3)])
    bg = corners.mean(0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(-1))

    # Pixels near the background color are leftover background -> transparent.
    arr[..., 3][dist < bg_dist] = 0.0

    # Keep only the head (largest connected component), drop specks.
    labeled = measure.label(arr[..., 3] > 0.5, connectivity=2)
    biggest = 0
    biggest_area = 0
    for region in measure.regionprops(labeled):
        if region.area >= min_area and region.area > biggest_area:
            biggest_area = region.area
            biggest = region.label
    keep = (labeled == biggest).astype(np.float32) if biggest else np.zeros_like(arr[..., 3])
    arr[..., 3] *= keep

    return Image.fromarray(np.uint8(arr * 255), "RGBA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", default=DEFAULT_SRC)
    parser.add_argument("dst", nargs="?", default=DEFAULT_DST)
    parser.add_argument("--bg-dist", type=float, default=0.12,
                        help="color distance to background to treat as background")
    parser.add_argument("--min-area", type=int, default=200,
                        help="drop connected components smaller than this (px)")
    args = parser.parse_args()

    with Image.open(args.src) as img:
        matte = remove(img)
    out = color_key(matte, bg_dist=args.bg_dist, min_area=args.min_area)
    out.save(args.dst)
    print(f"saved {args.dst} ({out.mode}, {out.size})")


if __name__ == "__main__":
    main()