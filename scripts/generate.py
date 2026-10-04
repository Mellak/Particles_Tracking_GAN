"""Generate positron paths with a trained generator.

Conditions (initial energy, number of interactions, initial direction) come from one of:

  * --data-dir    GATE phase-space files positrons_<k>.npy. The conditions of those events are reused
                  and the GATE paths are loaded too, so that generated and GATE paths can be compared.
  * (default)     "demo" conditions: an analytic F-18 beta+ spectrum, a stand-in energy-to-interactions
                  rule (tuned on the thesis R_mean/R_max) and isotropic directions. For looking at paths, not an independent check of the
                  generators (see README, "Generate paths").

Output: a .npz file with the generated paths, shape (events, steps, 4) with columns (energy, x, y, z)
in (MeV, mm, mm, mm), zero-padded after the last interaction.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from src.models import GeneratorNumIntEnergyDirection2
from src.sampling import sample_conditions
from src.utils import (make_values_zero, plot_final_points, plot_generated_paths,
                       plot_x_distribution, plot_y_distribution, plot_z_distribution)

LATENT_DIM = 100
PSF_RANGE_MM = {"Water": (-2, 2), "RibBone": (-1.5, 1.5), "Lung": (-6, 6)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", default="Water", help="Water, RibBone or Lung (selects the default weights file).")
    p.add_argument("--emitter", default="F18", choices=["F18", "Ga68"])
    p.add_argument("--weights", default=None,
                   help="Generator weights. Default: weights/G_<emitter>_<material>.pth in this repository.")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Path length the generator was trained with (F18: 18, Ga68: 30).")
    p.add_argument("--num-paths", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=20000)
    p.add_argument("--data-dir", default=None, help="GATE data folder; if set, its conditions and paths are used.")
    p.add_argument("--min-selection", type=int, default=0, help="Use files positrons_<k>.npy with k > MIN_SELECTION ...")
    p.add_argument("--max-selection", type=int, default=20, help="... and k < MAX_SELECTION.")
    p.add_argument("--output", default=None, help="Output .npz. Default: generated_<emitter>_<material>.npz")
    p.add_argument("--plot", action="store_true", help="Show path and PSF plots (needs a display).")
    p.add_argument("--device", default=None, help="cuda or cpu. Default: cuda if available.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_generator(weights, device, num_steps):
    model = GeneratorNumIntEnergyDirection2(seq_len=num_steps).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def load_gate_conditions(data_dir, min_selection, max_selection, num_paths, num_steps):
    from src.dataloader import FastDataloader
    files = [np.load(os.path.join(data_dir, f)) for f in sorted(os.listdir(data_dir))
             if "positrons_" in f and min_selection < int(f.split("_")[1].split(".")[0]) < max_selection]
    dataset = FastDataloader(np.concatenate(files)[:num_paths], num_steps=num_steps)
    real_paths = dataset.data[:, :, 0, :].transpose(0, 2, 1)  # (events, steps, 4)
    return dataset.Energies, np.asarray(dataset.numInteractions), dataset.normalized_vector, real_paths


@torch.no_grad()
def generate(generator, energy, num_inter, direction, num_steps, batch_size, device):
    out = []
    t0 = time.time()
    for start in range(0, len(energy), batch_size):
        sl = slice(start, start + batch_size)
        e = torch.as_tensor(energy[sl], dtype=torch.float32, device=device)
        n = torch.as_tensor(num_inter[sl], dtype=torch.long, device=device)
        v = torch.as_tensor(direction[sl], dtype=torch.float32, device=device)
        masks = (torch.arange(num_steps, device=device)[None, :] < n[:, None]).float()
        z = torch.randn(len(n), LATENT_DIM, device=device)
        out.append(generator(z, n, e, masks, v).cpu())
    print(f"Generated {len(energy)} paths in {time.time() - t0:.2f} s on {device}")
    paths = torch.cat(out)[:, :, 0, :].transpose(1, 2).numpy()  # (events, steps, 4)
    return make_values_zero(paths, num_inter)


def end_point_radius(paths, num_inter):
    """Distance in mm from the origin to the last interaction of each path."""
    end = paths[np.arange(len(paths)), np.asarray(num_inter) - 1, 1:]
    return np.linalg.norm(end, axis=1)


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_steps = args.num_steps or (30 if args.emitter == "Ga68" else 18)
    weights = args.weights or os.path.join(REPO_ROOT, "weights", f"G_{args.emitter}_{args.material}.pth")
    output = args.output or f"generated_{args.emitter}_{args.material}.npz"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    real_paths = None
    if args.data_dir:
        energy, num_inter, direction, real_paths = load_gate_conditions(
            args.data_dir, args.min_selection, args.max_selection, args.num_paths, num_steps)
    else:
        if args.emitter != "F18":
            sys.exit("Demo conditions are only implemented for F18. Pass --data-dir for other emitters.")
        print("WARNING: using demo conditions (analytic F-18 spectrum, stand-in energy-to-interactions "
              "rule). Pass --data-dir with GATE data for conditions that match the training data.")
        energy, num_inter, direction = sample_conditions(args.num_paths, rng, args.material)

    generator = load_generator(weights, device, num_steps)
    paths = generate(generator, energy, num_inter, direction, num_steps, args.batch_size, device)
    np.savez_compressed(output, paths=paths, num_interactions=num_inter, energy=energy, direction=direction)
    print("Saved", output)

    g_r = end_point_radius(paths, num_inter)
    print(f"Generated  R_mean = {g_r.mean():.3f} mm   R_max = {g_r.max():.3f} mm")
    if real_paths is not None:
        r_r = end_point_radius(real_paths, num_inter)
        print(f"GATE       R_mean = {r_r.mean():.3f} mm   R_max = {r_r.max():.3f} mm")

    if args.plot:
        lo, hi = PSF_RANGE_MM.get(args.material, (-2, 2))
        if real_paths is not None:
            plot_x_distribution(paths, real_paths, min_x=lo, max_x=hi, num_bins=101)
            plot_y_distribution(paths, real_paths, min_x=lo, max_x=hi, num_bins=101)
            plot_z_distribution(paths, real_paths, min_x=lo, max_x=hi, num_bins=101)
        ref = real_paths[:100] if real_paths is not None else paths[:0]
        plot_generated_paths(paths[:100], ref)
        plot_final_points(paths[:100], ref)


if __name__ == "__main__":
    main()
