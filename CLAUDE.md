# NeuroChain: working rules for Claude sessions

NeuroChain is a portfolio-grade neurotech engineering project by a 2nd-year
Electrical Engineering student at UNSW. Build each unit to a finished, polished
standard. The owner learns by reverse-engineering complete work afterwards, so
the reasoning has to be written down, not just the code.

Plan, primer, skills matrix and frontier analysis:
https://claude.ai/artifact/JApV8T8qvqvoyqATbQ3LVF (a Claude Doc; read it with the
Docs connector, not a web fetch).

## Research question

What is the lowest-power neural sensing chain (µW per channel) that still decodes
well while the implant is stimulating and its electrodes are aging?

- **A (deep, the spine):** co-design front end + ADC + features + decoder under a
  power model; produce an accuracy-vs-µW Pareto surface; implement the winning
  feature path in RTL.
- **B (deep):** record while stimulating: artifact propagation, front-end
  saturation and recovery, charge-balanced stimulation, fixed-point adaptive
  artifact cancellation, then RTL.
- **C (thin evaluation module):** simulated electrode aging (impedance, noise,
  amplitude drift) plus held-out-session tests on the public NoMAD dataset. A
  stress test only.

## Owner context

- Courses done: ELEC2133 (analog), ELEC3104 (DSP), ELEC2141 (digital), MATH2099
  (linear algebra / statistics). Explain new ideas in terms of these.
- Target roles: neurotech startups (Neuralink, Paradromics, Synchron, Precision,
  Axo).
- Machine: MacBook M5 Pro, Apple Silicon. Use PyTorch MPS where useful.
- Prefer open-source, free tools.
- No FPGA board yet: synthesise for Lattice ECP5 (LFE5U-25F) with Yosys +
  nextpnr, no hardware in the loop.
- The hardware phase happens later at a makerspace. Always choose the cheapest
  viable parts.
- Safety: saline only, never a person. Battery or isolated USB for any on-body
  recording.

## Non-negotiable rules

1. **Never invent results.** Every number in the README comes from code in this
   repo. README tables are generated from `results/*.json` by
   `scripts/update_readme.py`; do not hand-edit text between the
   `<!-- BEGIN/END ... -->` markers. If something fails or underperforms, report
   it honestly, in the README and in the unit summary.
2. **Never commit data.** `data/` is git-ignored. Datasets arrive through
   `neurochain.data.download` (`make data`), verified by SHA-256.
3. **One command per figure.** Every figure is reproducible from a `make`
   target. Each figure states its dataset, split and units on the figure itself.
4. **Fixed random seeds.** Anything stochastic takes an explicit `seed` and uses
   `numpy.random.default_rng(seed)`. No global RNG state.
5. **Work unit by unit, in order** (U0 to U8, see `TODO.md`). A unit ends at its
   done-criterion, not on a date.

## Quality bar

- `src/` layout package `neurochain`, type hints and docstrings on every public
  function, pytest for every module.
- `ruff` (lint + format) through pre-commit, `mypy` on `src/`, GitHub Actions CI
  running lint, types and tests.
- Tests must not need downloaded data. Use `neurochain.data.synthetic` or tiny
  fixtures built in `tests/`. Tests that need real data are marked
  `@pytest.mark.data` and skip when the data is absent.
- Units go in names or docstrings (`fs_hz`, `_uv`, `_lsb`). Never leave a
  physical quantity unitless.

## Learning artifacts (required for every unit)

1. `docs/Ux_design.md`: the EE reasoning, alternatives considered, why each
   choice won, and which UNSW course concept it uses.
2. A test or script that prints the unit's key number (`make uX-key`).
3. Three prediction checks: a parameter to change plus the question "what do you
   expect to happen?", with the answer in a collapsed `<details>` section. The
   answers must be numbers measured by running the code, not guesses.

## After each unit

1. Commit (conventional message, e.g. `U0: repo, loaders, band-split`).
2. Update `README.md` (results table, figures) and `TODO.md`.
3. Give the owner a short summary: key number, what was surprising, what to read
   next.
4. Cite every paper relied on in `docs/references.md`, with a link.

## Commands

```
make setup     # uv sync + pre-commit hooks
make data      # download / verify datasets
make data-describe  # print what the Nason and NoMAD files contain
make u0        # data + U0 figures + key number + README table
make u1        # U1 figures, metrics, netlists, README table (needs ngspice + Nason data)
make u1-key    # U1 key numbers; ARGS="--bias-scale 0.5" changes one parameter
make test      # pytest
make lint      # ruff + mypy
```

Run Python through uv: `uv run python ...`, `uv run pytest`.

## Data access facts (checked 2026-11-13)

- **Neuralink compression challenge:** the original bucket
  (`content.neuralink.com/compression-challenge/`) returns 404. The downloader
  falls back to the Internet Archive snapshot of `data.zip` and pins its
  SHA-256. 743 mono 16-bit WAV clips, 19 531 Hz, about 5 s each. No voltage
  scale is published, so amplitudes are reported in ADC LSB and in multiples of
  the noise σ, never in µV.
- **NoMAD (Dryad, doi:10.5061/dryad.q83bk3jtp):** two NWB files (`session1.nwb`,
  `session2.nwb`, 247 MB total). File downloads need a Dryad bearer token
  (`DRYAD_TOKEN`) or a manual browser download; checksums come from the Dryad
  API. Verified by opening the files: one human participant (BrainGate T11,
  *Homo sapiens*), 192 channels, threshold crossings in 20 ms bins plus cursor
  kinematics. No broadband and no electrode table. `session2` was recorded
  first; `session1` follows 273.9 days later. Loader: `neurochain.data.nomad`.
- **Nason SBP data (Deep Blue, doi:10.7302/wwya-5q86):** three MATLAB v5 `.mat`
  files plus a README, 74 MB, CC BY-NC 4.0. Deep Blue sits behind a Cloudflare
  challenge, so it is a manual browser download, verified against hashes pinned
  from the first copy. Do not try to script around the challenge. Only
  `MultiUnitInVivo.mat` has broadband (10 s, one channel, 30 kHz, in µV); the
  online file is 96 channels of 1 ms SBP and threshold crossings with finger
  position. Loader: `neurochain.data.nason`.
- **MEArec:** generated locally (U2).

## Modelling conventions (set in U1)

- **Units:** from U1 on, work in real microvolts. The Neuralink data stays in
  ADC LSB and cannot go through the front-end emulator, which refuses anything
  not in µV.
- **"Front-end noise" means** noise integrated over 300 Hz to 7.5 kHz, referred
  to the tissue signal (SPICE's `inoise`). The NEF uses a different number
  (all frequencies), and a filtered recording gives a third. `docs/U1_design.md`
  section 3.3 has the table. Say which one a number is.
- **Electrode:** Randles with a constant-phase double layer (Straka et al.,
  2018), never the ideal-capacitor version, which understates noise 2.6 times.
- **Filters in anything implant-realistic are causal.** The emulator's are.
- **SPICE:** netlists are generated by `neurochain.frontend.spice`; the copies
  in `spice/` are written by `make u1` and a test keeps them in sync. Tests that
  need ngspice skip without it.

## Toolchain status

Installed: uv, git, gh, make, ngspice 47. Not yet installed (install when the
unit needs them, via Homebrew): verilator, yosys, nextpnr-ecp5 (U7), KiCad (U8).
