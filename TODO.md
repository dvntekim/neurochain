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

- [ ] **Nason data:** download the three `.mat` files by browser into
      `data/raw/nason/`, then write and test the loader, and pin the SHA-256
      values in `registry.py`. Needed before U2's second done-criterion.
- [ ] **NoMAD data:** download `session1.nwb` and `session2.nwb` into
      `data/raw/nomad/` (browser, or `DRYAD_TOKEN`), then write and test the NWB
      loader. Needed before U4.
- [ ] **NoMAD description to check:** the plan calls it a 192-channel human
      dataset. Dryad's abstract describes monkey motor cortex, and the release
      holds two sessions. Confirm species, channel count and session structure
      when the files are open; with two sessions, "held-out session" means train
      on one and test on the other, and U6's "R² vs simulated weeks" has to come
      from the aging model, not from many recorded weeks.
- [ ] **CI has not run on GitHub yet:** the workflow is written and its steps
      pass locally, but the repo has no remote.
- [ ] **Causal filtering:** U0's figures use zero-phase filters. A causal filter
      of the same order loses 60 % of threshold crossings. U2 onward must use
      causal filters for anything described as implant-realistic.

## U1: electrode and front-end model

Done-criterion: simulated noise within 10 % of hand calculation.

- [ ] Install ngspice (`brew install ngspice`)
- [ ] Randles electrode model in ngspice; impedance sweep against frequency
- [ ] LNA model; noise budget and NEF
- [ ] AC versus DC coupling; DC-offset tolerance
- [ ] Python front-end emulator applying modelled noise, bandwidth and offset
      to real data
- [ ] `docs/U1_design.md`, key-number script, three prediction checks

## U2: ADC and features

Done-criteria: detection F1 on MEArec across noise levels; SBP versus threshold
crossings reproduced on Nason data.

- [ ] MEArec recordings with ground-truth spikes
- [ ] Bit-depth and sample-rate sweeps
- [ ] Threshold, NEO and spiking-band-power paths, with causal filters
- [ ] Fixed-point versions of each path
- [ ] Nason loader and SBP-versus-threshold reproduction
- [ ] `docs/U2_design.md`, key-number script, three prediction checks

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
