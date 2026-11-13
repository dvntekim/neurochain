# References

Papers, datasets and tools this project relies on, grouped by the unit that
first uses them. DOIs were resolved on 2026-11-13.

## Datasets

- **Neuralink Compression Challenge.** One hour of raw N1 electrode recordings
  from non-human primate motor cortex. Original page (now returns 404):
  <https://content.neuralink.com/compression-challenge/README.html>. Archived
  copy: <https://web.archive.org/web/20250901131704/https://content.neuralink.com/compression-challenge/README.html>.
  Used in U0.
- **Nason, S. R. et al. (2021).** Example Data for "A Low-Power Band of Neuronal
  Spiking Activity Dominated by Local Single Units Improves the Performance of
  Brain-Machine Interfaces". University of Michigan, Deep Blue Data.
  <https://doi.org/10.7302/wwya-5q86>. CC BY-NC 4.0. Planned for U2 and U4.
- **Karpowicz, B. M. et al. (2025).** Data from: Stabilizing brain-computer
  interfaces through alignment of latent dynamics. Dryad.
  <https://doi.org/10.5061/dryad.q83bk3jtp>. CC0. Planned for U4 and U6.

## U0: data and band-split

- **Quiroga, R. Q., Nadasdy, Z. and Ben-Shaul, Y. (2004).** Unsupervised spike
  detection and sorting with wavelets and superparamagnetic clustering. *Neural
  Computation* 16(8), 1661-1687. <https://doi.org/10.1162/089976604774201631>.
  Source of the median-based noise estimate, σ = median(|x|)/0.6745.
- **Welch, P. D. (1967).** The use of fast Fourier transform for the estimation
  of power spectra: a method based on time averaging over short, modified
  periodograms. *IEEE Transactions on Audio and Electroacoustics* 15(2), 70-73.
  <https://doi.org/10.1109/TAU.1967.1161901>. The PSD estimator.
- **Rice, S. O. (1944).** Mathematical analysis of random noise. *Bell System
  Technical Journal* 23(3), 282-332.
  <https://doi.org/10.1002/j.1538-7305.1944.tb00874.x>. The continuous-time
  level-crossing rate; U0 explains why it overestimates for a sampled detector.
- **Christie, B. P. et al. (2015).** Comparison of spike sorting and
  thresholding of voltage waveforms for intracortical brain-machine interface
  performance. *Journal of Neural Engineering* 12(1), 016009.
  <https://doi.org/10.1088/1741-2560/12/1/016009>. Why threshold crossings are
  an acceptable substitute for spike sorting in decoding.
- **Nason, S. R. et al. (2020).** A low-power band of neuronal spiking activity
  dominated by local single units improves the performance of brain-machine
  interfaces. *Nature Biomedical Engineering* 4, 973-983.
  <https://doi.org/10.1038/s41551-020-0591-0>. The 300 to 1000 Hz spiking-band
  power feature (prediction check 4; reproduced in U2).
- **Marblestone, A. H. et al. (2013).** Physical principles for scalable neural
  recording. <https://arxiv.org/abs/1306.5709>. Background for the LFP and
  spike band definitions.
- **Virtanen, P. et al. (2020).** SciPy 1.0: fundamental algorithms for
  scientific computing in Python. *Nature Methods* 17, 261-272.
  <https://doi.org/10.1038/s41592-019-0686-2>. Filter design, filtering and
  Welch's method.

## Later units (listed in the plan, not yet relied on)

- **Karpowicz, B. M. et al. (2025).** Stabilizing brain-computer interfaces
  through alignment of latent dynamics. *Nature Communications* 16, 4662.
  <https://doi.org/10.1038/s41467-025-59652-y>. U6.
- **Harrison, R. R. and Charles, C. (2003).** A low-power low-noise CMOS
  amplifier for neural recording applications. *IEEE Journal of Solid-State
  Circuits* 38(6), 958-965. <https://collections.lib.utah.edu/ark:/87278/s68p6j6g>.
  U1.
