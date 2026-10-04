"""Render a looping GIF of generated F-18 positron paths being traced, in water and in bone.

Uses the released generators on CPU with the demo conditions of src/sampling.py (see README).
Usage: python scripts/make_animation.py --output figures/positron_paths.gif
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.models import GeneratorNumIntEnergyDirection2
from src.sampling import sample_conditions

BG, FG = "#0e1117", "#d8dee9"


def make_paths(material, n, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    g = GeneratorNumIntEnergyDirection2(seq_len=18)
    g.load_state_dict(torch.load(os.path.join(REPO_ROOT, "weights", f"G_F18_{material}.pth"), map_location="cpu"))
    g.eval()
    energy, num_inter, direction = sample_conditions(n, rng, material)
    e = torch.as_tensor(energy, dtype=torch.float32)
    k = torch.as_tensor(num_inter)
    masks = (torch.arange(18)[None] < k[:, None]).float()
    with torch.no_grad():
        out = g(torch.randn(n, 100), k, e, masks, torch.as_tensor(direction, dtype=torch.float32))
    paths = out[:, :, 0, :].transpose(1, 2).numpy()
    return paths[:, :, 1:], num_inter, energy  # xyz, N, E


def partial_path(xyz, n, t):
    """Points of one path traced up to fractional step t in [0, n-1]."""
    full = int(np.floor(t))
    pts = xyz[: full + 1]
    if full < n - 1:
        frac = t - full
        pts = np.vstack([pts, xyz[full] + frac * (xyz[full + 1] - xyz[full])])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "figures", "positron_paths.gif"))
    ap.add_argument("--num-paths", type=int, default=200)
    ap.add_argument("--trace-frames", type=int, default=34)
    ap.add_argument("--hold-frames", type=int, default=14)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    panels = [("Water", "Water", 1.6), ("RibBone", "Bone", 0.8)]
    data = {m: make_paths(m, a.num_paths, a.seed) for m, _, _ in panels}
    total = a.trace_frames + a.hold_frames
    frames = []
    for f in range(total):
        fig = plt.figure(figsize=(6.4, 3.4), dpi=100, facecolor=BG)
        prog = min(f, a.trace_frames - 1) / (a.trace_frames - 1)
        for i, (m, title, lim) in enumerate(panels):
            xyz, num_inter, energy = data[m]
            ax = fig.add_subplot(1, 2, i + 1, projection="3d", facecolor=BG)
            for p, n, en in zip(xyz, num_inter, energy):
                pts = partial_path(p, n, prog * (n - 1))
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=0.7, alpha=0.85, color=cm.plasma(en / 0.6335))
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1), zoom=1.55)
            ax.view_init(elev=20, azim=30 + 90 * f / total)
            ax.set_axis_off()
            ax.set_title(f"{title}  (axes ±{lim:g} mm)", color=FG, fontsize=9, pad=-2)
        fig.text(0.5, 0.03, f"{a.num_paths} generated F-18 positron paths per material (colour: initial energy)",
                 ha="center", color=FG, fontsize=7)
        fig.subplots_adjust(left=0, right=1, bottom=0.08, top=0.95, wspace=0)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        plt.close(fig)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    pal = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(a.output, save_all=True, append_images=q[1:], duration=70, loop=0, optimize=True, disposal=1)
    print(a.output, os.path.getsize(a.output) / 1e6, "MB", len(q), "frames")


if __name__ == "__main__":
    main()
