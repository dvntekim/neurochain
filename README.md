# NeuroChain

**What is the lowest-power neural sensing chain, in µW per channel, that still
decodes well while the implant is stimulating and its electrodes are aging?**

A brain implant has microvolt signals, a few milliwatts of power and a radio
that can send well under 1 % of what its electrodes record. Every stage of the
chain (electrode, amplifier, ADC, feature extraction, decoder) trades power
against decoding accuracy. NeuroChain models the whole chain, measures that
trade on public recordings, and implements the winning feature path in RTL.

The project has three parts:

- **A, co-design (the spine).** Front end, ADC, features and decoder under one
  power model, giving an accuracy-versus-µW Pareto surface.
- **B, recording while stimulating.** Stimulation artifact propagation,
  front-end saturation and recovery, and fixed-point adaptive cancellation.
- **C, aging stress test.** Simulated electrode aging on held-out sessions.

## Status

| Unit | What it delivers | State |
|---|---|---|
| U0 | Repo, data loaders, band-split, spectra and spike amplitudes | **Done** |
| U1 | Electrode and front-end model (ngspice), noise budget, NEF | Next |
| U2 | ADC sweeps; threshold, NEO and spiking-band-power features; fixed point | Planned |
| U3 | Power model and co-design: accuracy vs µW Pareto | Planned |
| U4 | Kalman and GRU decoders, session-held-out evaluation | Planned |
| U5 | Stimulation artifacts and fixed-point cancellation | Planned |
| U6 | Electrode aging stress test | Planned |
| U7 | RTL (SystemVerilog), bit-exact tests, ECP5 synthesis | Planned |
| U8 | ADS1299 board, saline phantom, stimulator: design for later build | Planned |

Done-criteria for each unit are in [TODO.md](TODO.md).

## Results so far

### U0: what real implant recordings look like

Dataset: Neuralink Compression Challenge, 743 five-second clips from an N1
implant in non-human primate motor cortex. There is no train/test split because
nothing is fitted; these are descriptive statistics over every clean clip.
Amplitudes are in ADC LSB because Neuralink does not publish the gain.

<!-- BEGIN U0_RESULTS -->
| Quantity | Value | How it is defined |
|---|---|---|
| Clips analysed | 737 of 743 | 6 excluded: more than 0.1% of samples on an ADC rail |
| Recording time analysed | 3,727 s | 19,531 Hz |
| Spike-band noise σ | 7.01 (IQR 6.07 to 8.00) LSB | MAD estimate per clip, 300 Hz to 7.5 kHz |
| Noise σ ÷ ideal quantisation noise | 24.3× | quantisation noise of a 1 LSB step is 1/√12 LSB rms |
| Share of power below 300 Hz | 83.3% (IQR 74.2% to 90.2%) | LFP band power ÷ (LFP + spike band power), from 1 Hz |
| PSD slope | f^-1.36 and f^-1.27 | 10 to 300 Hz, and 300 Hz to 5 kHz; median spectrum |
| Spikes detected at −4.5σ | 28,575 | 7.67 /s pooled; median clip 0.99 /s; 158 clips have none |
| Detections expected from Gaussian noise alone | 0.061 /s | false-alarm floor of the detector at this threshold |
| Median spike trough depth | 43.7 (IQR 36.4 to 55.8) LSB | = 5.55 σ; measured on the filtered spike band |
| Events whose larger phase is negative | 86.0% | two opposite-sign crossings within 1 ms count as one event |
<!-- END U0_RESULTS -->

This table is generated from [results/u0_metrics.json](results/u0_metrics.json)
by `make readme`; a test recomputes those metrics from the raw clips.

Three findings shape the next units:

1. **Five sixths of the power is below 300 Hz.** A spike-only chain can remove
   it before the ADC and spend the ADC's range on spikes (the U1 coupling
   decision).
2. **The ADC's step is one seventh of the noise.** Discarding 3 of its 10 bits
   raises the noise σ by 4 % and loses 6.5 % of threshold crossings. Discarding
   5 breaks the detector (ten times the detections). U2 measures this properly
   against ground truth.
3. **Offline numbers flatter the implant.** Replacing the zero-phase analysis
   filter with a causal filter of the same order, which is what hardware can
   run, loses 60 % of threshold crossings, because phase distortion makes spike
   troughs shallower and most spikes sit just above the threshold.

Findings 2 and 3 come from `make u0-key ARGS="..."`; the exact commands are in
[docs/U0_design.md](docs/U0_design.md#5-prediction-checks).

![Band-split traces of one clip and the amplitude spectral density of all clean clips](figures/u0_spectra.png)

*Top: one example clip, raw and split into the two bands. Bottom: median
amplitude spectral density across 737 clips with the interquartile range. Units:
ADC LSB and LSB/√Hz.*

![Detected spikes on an example clip and the distribution of spike amplitudes](figures/u0_spike_amplitudes.png)

*Top: threshold crossings on the example clip's spike band. Bottom: aligned
waveforms, and the trough depth of all 28,575 detected spikes in LSB and in
multiples of each clip's noise σ. The example clip is the clearest one by a
fixed rule, not a typical one.*

**Limits of these numbers.** There is no ground truth, so a "spike" is a
threshold crossing. The amplitude distribution is truncated at the threshold.
The analog chain before the ADC is unknown. More in
[docs/U0_design.md](docs/U0_design.md#limitations).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and `make`. Python 3.12 is installed
by uv.

```
make setup     # environment and git hooks
make u0        # download data (65 MB), write both figures, metrics and the table above
make test      # 80+ tests; the ones needing data skip if it is absent
```

`make help` lists every target.

## Data

Data is never committed. `make data` downloads what can be scripted and
verifies every file by SHA-256.

| Dataset | Used in | How it arrives |
|---|---|---|
| [Neuralink Compression Challenge](https://web.archive.org/web/20250901131704/https://content.neuralink.com/compression-challenge/README.html) | U0, U2, U5, U7 | Scripted. Neuralink removed the originals, so the Internet Archive copy is used and its hash pinned |
| [Nason et al. spiking-band-power data](https://doi.org/10.7302/wwya-5q86) | U2, U4 | Browser download (the host blocks scripts), then `make data` verifies it |
| [NoMAD](https://doi.org/10.5061/dryad.q83bk3jtp) | U4, U6 | Browser download, or scripted with a free Dryad API token in `DRYAD_TOKEN` |
| MEArec synthetic recordings | U2 | Generated locally |

## Layout

```
src/neurochain/
  recording.py          the container every loader returns
  data/                 registry, verified downloader, loaders, synthetic generator
  dsp/                  band-split filters, spectra, spike detection
  analysis/             per-unit analyses and their figures
  viz.py                shared figure style
scripts/                one script per figure or key number
tests/                  pytest; no test needs downloaded data unless marked `data`
docs/                   one design note per unit, plus references
figures/  results/      committed outputs, regenerated by make
```

## Learning notes

Each unit has a design note explaining the engineering reasoning, the
alternatives considered and the course concepts involved, plus prediction
checks: change one parameter, predict the result, then run it.

- [U0: repo, data and band-split](docs/U0_design.md)
- [References](docs/references.md)

## Safety

The hardware phase uses saline only, never a person. Any on-body recording
would use a battery or isolated USB.
