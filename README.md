# Fast-Track of F-18 Positron Paths Simulations Using GANs

Code and trained generators for the ISBI 2024 paper
*Fast-Track of F-18 Positron Paths Simulations Using GANs*
(Y. Mellak, K. Chatzipapas, A. Bousse, C. Chez-Le Rest, D. Visvikis, J. Bert),
[DOI 10.1109/ISBI56570.2024.10635834](https://doi.org/10.1109/ISBI56570.2024.10635834),
[arXiv:2403.06307](https://arxiv.org/abs/2403.06307).

The method is also described, with extensions, in Chapter 4 (section 1) of the PhD thesis of Y. Mellak
([HAL tel-05465688](https://theses.hal.science/tel-05465688)). "The thesis" below refers to that chapter.

## Motivation

Monte Carlo (MC) simulation is the reference for modelling positron transport in PET. Its particle-tracking
stage, where each positron is followed step by step until annihilation, is costly, which limits its use
when many events or fast turnaround are needed. This repository explores a data-driven alternative: a
generative network that produces a complete positron trajectory (3D positions and remaining kinetic energy
at each interaction) in a homogeneous medium. Annihilation distributions are obtained by binning the end
points of the generated paths.

## Method

A positron path is a matrix of `N` interactions with 4 features each: remaining kinetic energy and
(x, y, z) position. Paths start at the origin; energy decreases along the path and is zero at the last
interaction.

- **Generator** (`src/models.py`, `GeneratorNumIntEnergyDirection2`): a transformer encoder. Its input is
  a random vector (N(0, 1), dimension 100) concatenated with the embedded number of interactions, a mask
  that is zero beyond that number, the initial energy and the initial direction. The input is mapped to a
  sequence, the transformer encoder produces an embedding per sequence point, and a 1x1 convolution maps
  the embeddings to the four data features.
- **Discriminator** (`ViTwMask2`): a ViT-style binary classifier where each path point is a patch. It
  receives the path together with the embedded initial energy and number of interactions, with padding
  masked out.
- **Conditioning**: initial energy, number of interactions (which sets the path length, 3 to 18 for F-18)
  and initial direction.
- **Loss**: least-squares GAN loss, plus (i) a cosine term between the direction of the first path segment
  and the requested direction, which gives control over the emission direction, and (ii) two energy
  regularisers that penalise negative energies and energies that increase along the path, weighted by
  0.005. See `scripts/train.py`.
- **Training data**: GATE MC simulations of a point source at the origin in a 50 cm radius sphere, with
  the F-18 spectrum, in water, lung and bone, about 10,000 events per material. A phase-space actor records
  each step. Each epoch the paths are augmented by random rotations (see `src/dataloader.py`).
  The GATE data are not distributed in this repository.

![Generated F-18 positron paths being traced in water and bone](figures/positron_paths.gif)

*200 generated F-18 positron paths per material, from the released generators run on CPU with the demo
conditions. Created with `scripts/make_animation.py`.*

## Install

```bash
git clone https://github.com/Mellak/Particles_Tracking_GAN.git
cd Particles_Tracking_GAN
pip install -r requirements.txt
```

The released generators run on CPU or GPU.

## Repository layout

```
src/        models.py, dataloader.py, utils.py, sampling.py
scripts/    train.py, generate.py
notebooks/  Inference.ipynb
weights/    G_F18_Water.pth, G_F18_RibBone.pth, G_F18_Lung.pth
figures/
```

## Train

Training needs GATE phase-space files named `positrons_<k>.npy`, each an array of shape
`(events, steps, 4)` with rows `(energy [MeV], x, y, z [mm])`, zero-padded after the last interaction
(at least one all-zero row per event).

```bash
python scripts/train.py --data-dir /path/to/WaterF18 --material Water --emitter F18 --output-dir runs
```

Defaults match the original script: batch size 30, generator learning rate 1e-4 (discriminator 3e-4),
Adam, 10,000 epochs, files with `0 < k < 20`. Checkpoints, per-epoch generator weights and
diagnostic figures are written under `runs/<emitter>/<material>/`; training resumes automatically from the
checkpoint. Run `python scripts/train.py --help` for all options.

## Generate paths

With the released F-18 weights:

```bash
python scripts/generate.py --material Water --num-paths 20000 --output water_paths.npz
```

`--material` is `Water`, `RibBone` or `Lung`. The output `.npz` holds `paths` of shape `(events, 18, 4)`
(energy, x, y, z; MeV and mm) and the conditions, and the script prints R_mean and R_max of the generated
end points.

The generator needs the conditions (initial energy, number of interactions, direction). Two sources:

- `--data-dir /path/to/WaterF18`: reuse the conditions of GATE events, and also load the GATE paths for
  comparison. This is the setting of the paper and the thesis.
- default ("demo" conditions): an analytic F-18 beta+ spectrum, isotropic directions, and a stand-in rule
  mapping energy to number of interactions (`src/sampling.py`). In the paper and thesis, the number of
  interactions comes from an energy-to-interactions histogram built from the GATE data, which is not
  distributed here. The stand-in rule has two coefficients per material, tuned so that the generated
  R_mean and R_max come out close to the F-18 values in the Results table. It is therefore not an
  independent check of the generators. Use `--data-dir` for any quantitative comparison
  (see [Limitations](#limitations)).

`notebooks/Inference.ipynb` is the original inference notebook, updated for the new layout. It needs GATE
data, like `--data-dir`.

## Results

All numbers below are from the thesis (chapter 4, section 1), with GATE as reference: mean (`R_mean`) and
maximum (`R_max`) radius of the end points of the paths.

| Material | | F-18 R_mean | F-18 R_max | Ga-68 R_mean | Ga-68 R_max |
|---|---|---|---|---|---|
| Water | GATE | 0.52 mm | 2.13 mm | 2.39 mm | 11.15 mm |
| | GAN | 0.52 mm | 2.02 mm | 2.37 mm | 11.07 mm |
| Bone | GATE | 0.25 mm | 1.07 mm | 1.28 mm | 4.78 mm |
| | GAN | 0.26 mm | 0.94 mm | 1.26 mm | 4.26 mm |
| Lung | GATE | 1.92 mm | 7.82 mm | 9.00 mm | 34.51 mm |
| | GAN | 1.93 mm | 7.60 mm | 8.22 mm | 29.24 mm |

R_mean agrees closely with GATE. R_max, a statistic of the tail, differs by up to 13% across all
materials and both isotopes (up to 12% for F-18, in bone). The 1D point spread functions along x, y and z
overlap closely with GATE for the three materials. The Ga-68 columns come from the thesis extension, not
from the weights in this repository (see [Thesis extensions](#thesis-extensions-not-all-in-this-repository)).

**Speed.** The paper reports about 6 s for 20,000 paths with the GAN (batch of 20,000 events) against about
45 s with GATE for the same setup (three point sources of 0.2 MBq in 5 cm radius spheres). The thesis
chapter does not repeat this comparison. These timings depend on hardware and on the GATE configuration
(physics list, cuts), and were not re-measured for this README. The generator is small (about 1 MB of
weights each) and can generate large batches in one shot.

## Limitations

These are the limitations stated in the thesis discussion, plus what is specific to this repository:

- One model is needed per radionuclide and per material, since the spectrum and the number of interactions
  depend on both.
- The generator needs the number of interactions as an input, to define the output length and the padding.
  It comes from a histogram over GATE data and is not available in this repository (hence the demo
  conditions above). This becomes problematic at material boundaries, where the number of interactions of
  the remaining track is inferred from the residual energy through the same histogram.
- Training data in voxelised or heterogeneous GATE volumes contain many extra steps near boundaries
  (hundreds of steps per track), which makes a single consistent model hard to train. The thesis proposes
  simplified MC tracks and an autoregressive model as future work.
- Diffusion-based training of the same generator (400 steps) was explored in the thesis; it removes the
  need for the number of interactions but multiplies generation time by 400.
- Material boundaries remain the most sensitive case. The models in this repository are for homogeneous
  media only.

## Thesis extensions (not all in this repository)

Chapter 4 of the thesis extends the paper in two ways. **Only the F-18 water, lung and bone generators are
in this repository.** The following is described in the thesis and is not included here:

- **Ga-68.** Generators for Ga-68, whose higher-energy positrons have up to 30 interactions per path
  (instead of 18), trained in water, lung and bone. The training code accepts `--emitter Ga68`
  (30 steps), but this path has not been tested here and no Ga-68 weights are provided. The Ga-68 numbers in
  the results table are from the thesis.
- **RGIMMT**, recursive generation for heterogeneous materials. For each material in a voxelised phantom,
  the number of positrons is computed from the activity. Each positron is generated with the generator of
  its material, conditioned on an energy sampled from the isotope spectrum, a number of interactions from
  the energy-to-interactions histogram, and an isotropic direction, then placed at an emission point in
  an active voxel. If the path crosses a material boundary, it is truncated at the interface, and the
  generator of the new material is called again from the state at the crossing (position, direction,
  residual energy). This repeats until the track ends inside a material, and the end point is binned in
  the annihilation volume. The thesis phantom is a rod of four spheres joined by a bridge (400x100x100
  voxels of 1 mm^3, 20 million events), spanning bone, water and lung. The thesis reports that, in
  such a phantom, the direct GAN overestimates the positron range when crossing from low-density
  to denser material, whereas RGIMMT stays close to GATE across the transitions. The RGIMMT code, the
  Ga-68 code and weights, the energy-to-interactions histograms and the phantom are not in this
  repository.

## Why this led to DDConv

The thesis explains why this particle-tracking approach was not used for positron range correction in image
reconstruction, which was its initial purpose. In its recursive form, the method is computationally heavy
in heterogeneous phantoms, where individual tracks can have hundreds of segments, and realistic scans
involve hundreds of millions of positrons, so runtime is prohibitive. In addition, iterative reconstruction
needs the transpose of the blur operator, which is not tractable for a path-wise simulation. The thesis
therefore treats the positron range effect at the image level, with learned spatial transformations,
whose cost depends on the image size rather than on the number of events. That work is the DDConv method:
[github.com/Mellak/ddconv-prc](https://github.com/Mellak/ddconv-prc) (code to be released there).

## Citation

```bibtex
@inproceedings{mellak2024fasttrack,
  title     = {Fast-Track of {F-18} Positron Paths Simulations Using {GANs}},
  author    = {Mellak, Youness and Chatzipapas, Konstantinos and Bousse, Alexandre and
               Chez-Le Rest, Catherine and Visvikis, Dimitris and Bert, Julien},
  booktitle = {2024 IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2024},
  doi       = {10.1109/ISBI56570.2024.10635834},
  eprint    = {2403.06307},
  archivePrefix = {arXiv}
}
```

## License

MIT, see [LICENSE](LICENSE).
