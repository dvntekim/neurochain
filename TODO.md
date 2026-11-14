# NeuroChain build units

Units are done in order. Each ends at its done-criterion, not on a date.
After each unit: commit, update README.md and this file, write the summary.

Every unit also ships its learning artifacts: `docs/Ux_design.md`, a script or
test that prints the key number, and three prediction checks with measured
answers.

## U0: repo and data — DONE

Done-criterion: spectra and spike-amplitude figures reproduce from one command.

- [x] Package layout (`src/neurochain`), uv environment, Makefile
- [x] ruff, mypy, pytest, pre-commit, GitHub Actions workflow
- [x] Dataset registry and verified downloader
- [x] Neuralink loader (int16 to 10-bit code, saturation flag)
- [x] Seeded synthetic recording with ground truth (test fixture)
- [x] Band-split: LFP below 300 Hz, spikes 300 Hz to 7.5 kHz
- [x] Welch spectra, MAD noise estimate, threshold spike detection
- [x] `make u0` writes `figures/u0_spectra.png`, `figures/u0_spike_amplitudes.png`,
      `results/u0_metrics.json` and the README table
- [x] `docs/U0_design.md` with four prediction checks; `make u0-key`

Key number: spike-band noise σ = 7.01 LSB; median detected spike 43.7 LSB = 5.55σ.

Carried forward from U0:

- [x] **Nason data:** files downloaded, SHA-256 pinned in `registry.py`, loader
      in `neurochain.data.nason` with tests (`tests/test_nason.py`, and
      `tests/test_nason_data.py` on the real files). Ready for U2.
- [x] **NoMAD data:** files downloaded and verified, loader in
      `neurochain.data.nomad` with tests (`tests/test_nomad.py`, and
      `tests/test_nomad_data.py` on the real files). Ready for U4.
- [x] **NoMAD description checked:** the plan was right. Both files record one
      human participant (BrainGate T11, *Homo sapiens*) with 192 channels.
      `make data-describe` prints this. The monkey wording comes from the
      paper's abstract, which Dryad reuses; this release is the human data.

What opening the files showed, for the units that use them:

- [ ] **U6 cannot age NoMAD waveforms, because there are none.** NoMAD holds
      threshold crossings in 20 ms bins: no broadband, no spike shapes, no
      electrode table, no impedance. Impedance, noise and amplitude drift
      therefore have to be modelled as changes to crossing counts (for example
      through an amplitude-distribution model fitted on Neuralink or Nason
      broadband), or the aging model has to be validated on broadband data and
      only its effect carried to NoMAD. Decide this in `docs/U6_design.md`.
- [ ] **NoMAD is two sessions 273.9 days apart, and `session2` is the earlier
      one.** Held-out-session evaluation means train on one, test on the other.
      U6's "R² vs simulated weeks" comes from the aging model; the real pair
      gives one measured point at about 39 weeks to compare it with.
- [ ] **NoMAD time is not continuous:** 255 contiguous segments in `session1`
      (some one bin long) and 16 in `session2`. Decoders and any filtering must
      respect `NomadSession.segment_starts`. Only about 37 % of bins fall
      inside a trial.
- [ ] **NoMAD `cursor_vel` is not the derivative of `cursor_pos`** (correlation
      about 0.1 to 0.2 with the finite difference). Work out what it is before
      choosing the decoding target in U4.
- [x] **Nason online SBP is a sum over a varying number of samples** (0 to 4 per
      1 ms bin, mean 2.0). `neurochain.analysis.u2.online_features` divides by
      the sample count in each 50 ms bin. Its unit is not stated.
- [x] **Nason in-vivo broadband is the only µV-scaled real recording so far**
      (10 s, one channel, 30 kHz, three sorted units). Used in U1: its
      spike-band noise is 13.2 µV, 4.4 times the front-end model's floor.
- [x] **CI on GitHub:** green on `github.com/dvntekim/neurochain`, including the
      U1 commit (d248678), where the ngspice tests ran on Linux with ngspice 45
      and passed.
- [ ] **CI housekeeping:** GitHub warns that `actions/checkout@v4` and
      `astral-sh/setup-uv@v6` target the deprecated Node 20 runtime. Bump both
      when newer majors are confirmed.
- [x] **Causal filtering:** U0's figures use zero-phase filters. A causal filter
      of the same order loses 60 % of threshold crossings. From U2 on every
      implant path is causal. U2 found the loss belonged to that 4th-order
      filter: with a 2nd-order high-pass, causal detection scores F1 0.63
      against 0.66 zero-phase.

## U1: electrode and front-end model — DONE

Done-criterion: simulated noise within 10 % of hand calculation.

- [x] ngspice installed (version 47); CI installs it too
- [x] Randles electrode in ngspice, with the measured constant-phase double
      layer as an RC ladder; impedance sweep against the analytic model
- [x] Capacitive-feedback amplifier model; noise budget and NEF
- [x] Hand calculation, analytic network model and ngspice agree within 0.3 %
      for three electrode models (`tests/test_frontend.py`)
- [x] AC versus DC coupling: offset tolerance, ADC bits, settling after a step
- [x] Causal, seeded front-end emulator; checked on silence and on the Nason
      in-vivo recording
- [x] `make u1` writes four figures, `results/u1_metrics.json`, the README
      table and the reference netlists in `spice/`
- [x] `docs/U1_design.md` with the worked hand calculation and four prediction
      checks; `make u1-key`

Key number: front-end noise floor 3.41 µV rms in the spike band (electrode
2.51 µV, amplifier 2.31 µV at 18 µW).

Carried forward from U1:

- [ ] **No public dataset here has both broadband and behaviour.** The emulator
      needs a signal in microvolts. Neuralink is broadband without a voltage
      scale or behaviour; Nason's behaviour file and NoMAD hold features, not
      broadband. So the front end cannot simply be swept in front of a decoder.
      U3 needs a deliberate bridge: measure how front-end settings change the
      features (threshold crossings, SBP) on broadband with ground truth
      (MEArec, Nason in vivo), then apply that change to the feature datasets.
      Decide this at the start of U3, or find a dataset that has both.
- [ ] **U3 must include neural background noise.** In vivo the noise is four
      times the front end's floor, and a quarter of the bias current costs 4 %
      in σ. A power model that sees only front-end noise would overspend on the
      amplifier.
- [x] **Aliasing belongs to U2's ADC model.** Done: the emulator's
      `alias_zones` folds amplifier noise from above the Nyquist frequency back
      in, and the ADC samples with no digital filter in front.
- [ ] **No 1/f noise in the amplifier model.** Small in the spike band (about
      4 % of noise power for a 100 Hz corner, hand estimate), large below
      300 Hz. Add it before any LFP-band result.
- [ ] **Rails are a hard clip.** U5 needs saturation recovery.
- [ ] **Harrison's supply voltage is unverified.** The 5 V that gives 16 µA and
      NEF 4.0 is from memory; the full text was not reachable. Check the paper.

## U2: ADC and features — DONE

Done-criteria: detection F1 on MEArec across noise levels; SBP versus threshold
crossings reproduced on Nason data.

- [x] MEArec recording with ground-truth spikes (`make mearec`; 32 channels,
      16 units, 60 s; the far-neuron background is kept separate so its level
      can be set)
- [x] ADC model: sample rate, bits and input range referred to the electrode;
      sampling aliases, noise from above Nyquist is folded in
- [x] Bit-depth and sample-rate sweeps, with bootstrap intervals over channels
- [x] Threshold and NEO detectors and spiking-band power, all causal
- [x] Fixed-point versions of each path, matching floating point
- [x] SBP versus threshold crossings reproduced on the Nason data (simulation,
      in vivo, and decoding from the online data)
- [x] `make u2` writes four figures, `results/u2_metrics.json` and the README
      table; `docs/U2_design.md` with four prediction checks; `make u2-key`

Key number: causal threshold detection at in-vivo noise (13 µV) scores F1 0.63
at 30 kS/s and 10 bits, and still 0.61 at 6 bits or at 10 kS/s.

Carried forward from U2:

- [ ] **The ADC's input range has to be chosen on real data.** MEArec has almost
      no LFP. The Nason broadband has an RMS of 972 µV, which would clip the
      ±1 mV range used here. U3 must choose range and analog high-pass corner
      together: U1 showed a 300 Hz analog corner costs 12 % in amplifier noise.
- [ ] **NEO was not retuned per ADC setting**, which is why its sweep curves are
      irregular. If U3 keeps NEO as a candidate, tune its factor at each sample
      rate and resolution. On F1 it has not beaten the plain threshold.
- [ ] **U3's bridge from front end to decoder** (see U1's carried-forward item):
      U2 gives the first half. Detection F1 and feature tracking are now known
      as functions of noise, bits and sample rate. The second half is turning
      those into a change in the Nason and NoMAD features.
- [ ] **RTL parameters are fixed** for U7: 14 fractional bits and 6 guard bits
      at 30 kS/s (29-bit accumulator); 8 and 2 at 2 kS/s (18-bit accumulator);
      round half up; numerator `b0*(1 - z^-1)^2`. The Python in
      `neurochain.fixedpoint` is the reference the RTL must match bit for bit.
- [ ] **MEArec with NEURON 9:** two legacy synapse mechanisms are patched in a
      copy under `data/mearec/` by `scripts/make_mearec.py`. If MEArec or NEURON
      change version, check the patch still applies.

## U3: power model and co-design (A)

Done-criterion: accuracy-versus-µW Pareto plot on held-out sessions.

- [ ] Per-stage power model: LNA, ADC, DSP, link at a stated pJ/bit
- [ ] Sweep or optimise configurations
- [ ] `docs/U3_design.md`, key-number script, three prediction checks

## U4: decoders and evaluation

Done-criterion: R² per configuration with confidence intervals.

- [ ] Kalman filter decoder
- [ ] Small GRU decoder (PyTorch, MPS)
- [ ] Session-held-out splits
- [ ] `docs/U4_design.md`, key-number script, three prediction checks

## U5: stimulation artifacts (B)

Done-criterion: decoding loss with and without cancellation, plus recovery time.

- [ ] Artifact model: dipole propagation, front-end saturation and recovery in SPICE
- [ ] Injection into real recordings
- [ ] Fixed-point adaptive cancellation
- [ ] `docs/U5_design.md`, key-number script, three prediction checks

## U6: aging stress test (C)

Done-criterion: R² versus simulated weeks.

- [ ] Encapsulation and impedance drift model applied to NoMAD data
- [ ] Comparison with a simple stabiliser
- [ ] `docs/U6_design.md`, key-number script, three prediction checks

## U7: RTL

Done-criterion: bit-exact on 10 minutes of real data; LUTs, fmax and estimated
µW per channel reported.

- [ ] Install verilator, yosys, nextpnr-ecp5
- [ ] SBP extractor and artifact canceller in SystemVerilog
- [ ] Verilator tests, bit-exact against the fixed-point Python
- [ ] Yosys and nextpnr for ECP5 (LFE5U-25F)
- [ ] `docs/U7_design.md`, key-number script, three prediction checks

## U8: hardware design (for later)

Done-criterion: schematic, layout, BOM sorted by cost, bring-up test plan.

- [ ] KiCad design: ADS1299 board with an STM32 over SPI
- [ ] Saline phantom with a DAC-driven playback electrode
- [ ] Charge-balanced current stimulator
- [ ] Saline only, never a person. Battery or isolated USB for any on-body recording
- [ ] `docs/U8_design.md`, three prediction checks
