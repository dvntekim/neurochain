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

## U1: electrode and front-end model

- **Straka, M. M., Shafer, B., Vasudevan, S., Welle, C. and Rieth, L. (2018).**
  Characterizing longitudinal changes in the impedance spectra of in-vivo
  peripheral nerve electrodes. *Micromachines* 9(11), 587.
  <https://doi.org/10.3390/mi9110587>. Open access. Table 1 gives the Randles
  fits used for the electrode: Blackrock platinum Utah arrays in 0.15 M PBS
  (96 spectra) and two or more weeks after implantation in rat sciatic nerve
  (854 spectra). Values were read from the article's full-text table.
- **Harrison, R. R. and Charles, C. (2003).** A low-power low-noise CMOS
  amplifier for neural recording applications. *IEEE Journal of Solid-State
  Circuits* 38(6), 958-965. <https://doi.org/10.1109/JSSC.2003.811979>. The
  amplifier topology. From the abstract: 0.025 Hz to 7.2 kHz, 2.2 µV rms, 80 µW.
  The gain-setting capacitors (20 pF and 200 fF) are as reported for the same
  design. **Not re-verified this session:** the ±2.5 V supply, which gives the
  16 µA and the NEF of 4.0 used here. The full text was not reachable; check it
  before quoting those three numbers elsewhere.
- **Steyaert, M. S. J. and Sansen, W. M. C. (1987).** A micropower low-noise
  monolithic instrumentation amplifier for medical purposes. *IEEE Journal of
  Solid-State Circuits* 22(6), 1163-1168.
  <https://doi.org/10.1109/JSSC.1987.1052869>. Origin of the noise efficiency
  factor.
- **Lempka, S. F. et al. (2011).** Theoretical analysis of intracortical
  microelectrode recordings. *Journal of Neural Engineering* 8(4), 045006.
  <https://doi.org/10.1088/1741-2560/8/4/045006> (open manuscript:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3196618>). Thermal noise of 6.2 µV
  and biological noise of 10.2 µV over 450 to 5000 Hz, 12 µV in total; brain
  resistivity 300 Ω·cm. Used to interpret why a real channel is noisier than the
  front-end model.
- **ngspice.** Open-source circuit simulator, version 47 used here.
  <https://ngspice.sourceforge.io/>.

## U2: ADC and features

- **Buccino, A. P. and Einevoll, G. T. (2021).** MEArec: a fast and customizable
  testbench simulator for ground-truth extracellular spiking activity.
  *Neuroinformatics* 19, 185-204. <https://doi.org/10.1007/s12021-020-09467-7>.
  The ground-truth recordings; version 1.11.0, with NEURON 9 and LFPy 2.3.
- **Mukhopadhyay, S. and Ray, G. C. (1998).** A new interpretation of nonlinear
  energy operator and its efficacy in spike detection. *IEEE Transactions on
  Biomedical Engineering* 45(2), 180-187. <https://doi.org/10.1109/10.661266>.
  The NEO detector and its threshold rule.
- **Nason, S. R. et al. (2020)** (listed under U0). The spiking-band power recipe
  and the data it is reproduced on.

## Later units (listed in the plan, not yet relied on)

- **Karpowicz, B. M. et al. (2025).** Stabilizing brain-computer interfaces
  through alignment of latent dynamics. *Nature Communications* 16, 4662.
  <https://doi.org/10.1038/s41467-025-59652-y>. U6.
