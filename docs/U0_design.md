# U0 design: repo, data and band-split

**Done-criterion:** spectra and spike-amplitude figures reproduce from one command.
**Status:** met. `make u0` downloads the data, writes both figures, the metrics
file and the README table.

Every number below comes from `results/u0_metrics.json` or from
`scripts/u0_key_number.py`. If you change the code, rerun `make u0` and the
numbers here may be stale; the README table never is, because it is generated.

## 1. What U0 has to answer

Before modelling a front end (U1) or sweeping ADC bits (U2), three facts about
real recordings are needed:

1. **Where is the power?** This sets what the front end must pass and reject.
2. **How big is the noise in ADC steps?** This says whether the ADC has bits to
   spare.
3. **How big are spikes compared with the noise?** This sets how much any later
   stage can degrade the signal before detection fails.

## 2. The data, and three things measured about it

The Neuralink Compression Challenge is 743 WAV clips of about 5 s each
(3,757 s in total), mono, 16-bit, 19,531 Hz, recorded from an N1 implant in
macaque motor cortex.

**The WAV is a stretched 10-bit signal.** Across all clips only 1,023 distinct
int16 values occur, 64 or 65 apart, spanning the full int16 range. So
`int16 >> 6` (an arithmetic shift, as in ELEC2141) recovers the signed 10-bit
ADC code in [-512, 511] with no loss: every value that occurs maps to its own
code. `tests/test_neuralink_data.py::test_shift_by_six_is_lossless` checks this
on the whole dataset.

**There is no voltage scale.** Neuralink does not publish the gain, so the
loader reports ADC LSB. I did not convert to microvolts, because that would
mean inventing the gain. Where a physical comparison helps, amplitudes are given
in multiples of the noise σ, which needs no gain.

**Some clips hit the rails.** Eight clips contain samples at code -512 or 511.
Six have more than 0.1 % of samples railed (five of them more than 38 %) and are
excluded from every statistic, leaving 737 clips and 3,727 s. A railed front end
is exactly the failure U5 studies, so the loader keeps the saturation fraction
in each recording's metadata.

**What is not known:** which electrode each clip came from, whether clips
overlap in time, and what analog filtering came before the ADC. Each clip is
treated as one anonymous channel.

## 3. Decisions

### 3.1 Band-split filter: 4th-order Butterworth, second-order sections, zero-phase

The split is LFP below 300 Hz and spikes from 300 Hz to 7.5 kHz. At 19,531 Hz
the Nyquist frequency is 9,766 Hz, so the 7.5 kHz edge fits.

| Option | For | Against | Verdict |
|---|---|---|---|
| Butterworth IIR | Maximally flat passband, so spike amplitudes are not rippled; a few biquads | Non-linear phase | **Chosen** |
| Chebyshev / elliptic IIR | Steeper for the same order | Passband ripple changes the amplitudes U0 exists to measure | Rejected |
| Bessel IIR | Near-constant group delay, preserves spike shape when causal | Very gradual roll-off, poor band separation | Revisit in U2 for the causal path |
| Linear-phase FIR | Exactly constant group delay | A 300 Hz edge at 19.5 kHz needs hundreds of taps: hundreds of multiplies per sample on an implant | Rejected for the implant path |

Two implementation choices matter as much as the filter family:

- **Second-order sections, not one transfer function.** A 300 Hz corner at
  19.5 kHz puts poles very close to z = 1. Expanding an 8th-order polynomial
  there loses precision; cascaded biquads do not. It is also the structure the
  RTL will use in U7.
- **Zero-phase (forward then backward) for analysis.** This cancels the phase
  response, so a spike's trough stays where it was. It also squares the
  magnitude response: the corner becomes -6 dB instead of -3 dB and the roll-off
  doubles. It needs the whole record, so **an implant cannot run it**. Section 5
  shows how much that matters.

*Course link, ELEC3104:* IIR design from an analog prototype, pole placement,
group delay, and why linear phase needs a symmetric FIR.

### 3.2 Spectrum: Welch's method, 1.2 Hz bins

One periodogram of a 5 s clip has fine resolution but each bin's standard
deviation equals its mean, which is useless for reading a spectrum's shape.
Welch's method cuts the clip into Hann-windowed segments of 16,384 samples
(0.84 s, so bins 1.19 Hz apart) with 50 % overlap, and averages about 11
periodograms. The figure then takes the median across 737 clips.

The trade is resolution against variance: longer segments resolve lower
frequencies but leave fewer to average. Multitaper estimation is the principled
alternative, and was not needed with this many clips.

The plot shows amplitude spectral density (the square root of the PSD), in
LSB/√Hz, because that is the unit amplifier noise is specified in and U1 will
compare against it.

*Course link, ELEC3104:* DFT resolution is fs/N; windowing trades main-lobe
width against leakage. *MATH2099:* averaging K independent estimates cuts the
variance by K.

### 3.3 Noise estimate: median absolute deviation, not RMS

σ = median(|x - median(x)|) / 0.6745. The constant is the 75th percentile of a
standard normal, so for Gaussian noise this equals the standard deviation.

RMS is the obvious estimator and the wrong one: spikes are part of the signal,
so on an active channel RMS rises with firing rate and a threshold based on it
rises too, hiding the spikes it should find. The median ignores the small
fraction of samples that spikes occupy. In `tests/test_spikes.py`, spikes on
0.2 % of samples more than double the RMS while the MAD estimate moves under
3 %.

*Course link, MATH2099:* a robust estimator, and quantiles of the normal
distribution.

### 3.4 Spike detection: -4.5σ threshold, trough alignment, 1 ms dead time

A spike is a sample that crosses -4.5σ. The detector then takes the most
negative sample in the next 0.5 ms as the trough, and ignores further crossings
for 1 ms so one spike is counted once.

- **Why threshold crossings, not spike sorting.** Sorting needs waveform
  templates and far more computation. Threshold crossings are what a low-power
  implant can afford, and decode about as well (Christie et al., 2015). This
  project is about the cheapest chain that still decodes.
- **Why negative.** Counting each event by its larger phase, 86.0 % of events
  are negative-going, as expected for electrodes near the cell body. Counting
  raw crossings of each sign would be wrong: a filtered spike has lobes of both
  signs, and a large negative spike also crosses the positive threshold.
- **Why 4.5.** It is a common convention, and Section 4 quantifies what it
  buys. It is a free parameter that U3 will sweep.

### 3.5 False-alarm floor: exact for a sampled detector, not Rice's formula

Noise alone crosses the threshold sometimes. The predicted rate is the floor
below which a detection rate means nothing.

My first version used Rice's formula, rate = f₀·exp(-k²/2), the standard result
for a continuous Gaussian process. A unit test that counted crossings in
simulated noise disagreed with it by a factor of 2.2. The formula was right and
my use of it was wrong: at k = 4.5 a noise excursion past the threshold lasts
less than one sample period, so a sampled detector misses about half of them.

The code now uses the exact discrete-time result. The detector fires when one
sample is beyond the threshold and the previous one is not:

rate = fs · [Q(k) - P(both beyond)]

where Q is the normal tail and the joint term is a bivariate normal with the
noise's lag-one correlation ρ. By the Wiener-Khinchin theorem ρ comes from the
measured spectrum: ρ = ∫S(f)·cos(2πf/fs) df / ∫S(f) df. The simulation test now
agrees within 8 %.

*Course link, ELEC3104:* autocorrelation and PSD are a Fourier pair; sampling
only sees the signal at the sample instants. *MATH2099:* tail probabilities and
the bivariate normal.

### 3.6 Engineering choices

- **Downloads are verified before they exist.** Bytes go to a `.part` file and
  are renamed only after the SHA-256 matches. Neuralink has removed the
  original files, so the downloader falls back to the Internet Archive copy and
  pins its hash.
- **Tests never need downloaded data.** `neurochain.data.synthetic` generates a
  seeded recording with known spike times. Tests that do use real data are
  marked `data` and skip without it.
- **The README cannot drift.** Its table is generated from the metrics file,
  and a test recomputes the metrics from the raw clips and compares.

## 4. What the numbers say

| Question | Answer |
|---|---|
| Where is the power? | 83.3 % below 300 Hz in the median clip (IQR 74.2 to 90.2 %). The spectrum falls as f^-1.36 from 10 to 300 Hz and f^-1.27 from 300 Hz to 5 kHz, then flattens near 0.033 LSB/√Hz above 8 kHz. |
| How big is the noise? | Spike-band σ = 7.01 LSB (IQR 6.07 to 8.00). That is 24.3 times the 1/√12 LSB quantisation noise of the ADC. |
| How big are spikes? | Median trough 43.7 LSB (IQR 36.4 to 55.8), which is 5.55σ. |
| How many? | 28,575 at -4.5σ: 7.67 /s pooled, but 0.99 /s in the median clip, and 158 clips have none. |
| How many are noise? | Gaussian noise alone predicts 0.061 /s: under 1 % of the pooled detections, but about 6 % in the median clip. |

Three consequences for later units:

1. **The front end must handle a signal that is mostly not spikes.** Five sixths
   of the power is LFP. A spike-only chain can high-pass it away before the ADC
   and use the ADC's range on spikes. That is the U1 coupling decision.
2. **The 10-bit ADC looks over-specified for spike detection.** The noise is 7
   steps wide. Section 5 tests what happens with fewer bits. U2 does this
   properly, on data with ground truth.
3. **Most detected spikes are close to the threshold.** Half lie between 4.5σ
   and 5.55σ. A chain that loses even 20 % of spike amplitude loses a large
   share of detections. Section 5 shows a causal filter does exactly that.

### Limitations

- The amplitude distribution is truncated at the threshold. "Median spike
  5.55σ" describes detected spikes, not all spikes.
- Trough depth is measured after filtering. On the example clip the raw trough
  is about -158 LSB and the filtered one -132 LSB.
- The example clip in both figures was picked by rule as the clean clip with
  the highest median spike SNR. It is the clearest clip, not a typical one.
- There is no ground truth. Nothing here says a detection is a real spike. U2
  measures detection F1 on synthetic data where the truth is known.
- The median spectrum has a small bump near 40 Hz. I do not know its cause.
- The recording chain before the ADC is unknown, so the spectrum's shape mixes
  brain, electrode and Neuralink's analog filtering.

## 5. Prediction checks

For each one: read the question, write down what you expect and roughly how
much, then run the command and open the answer. The baseline is 28,575 spikes,
σ = 7.01 LSB, median trough 43.7 LSB.

### Check 1: lower the threshold from 4.5σ to 3.5σ

```
make u0-key ARGS="--threshold-sigma 3.5"
```

What happens to the number of detections, and how many of the new ones are
real?

<details>
<summary>Answer</summary>

Detections rise from 28,575 to 76,163 (2.7 times). The noise-only prediction
rises from 0.061 /s to 3.77 /s, a factor of 62, because the Gaussian tail is so
steep: one σ of threshold changes the false-alarm rate by nearly two orders of
magnitude.

Over 3,727 s, noise alone accounts for about 14,000 detections at 3.5σ against
about 230 at 4.5σ. So roughly 29 % of the 47,588 extra detections are expected
to be noise, and the rest are small real deflections (distant neurons). Every
clip now has at least one detection, and the median clip goes from 0.99 /s to
9.89 /s.

The lesson: a threshold detector's operating point sits on a very steep curve.
U3 treats the threshold as a design variable for that reason.
</details>

### Check 2: make the filter causal, as an implant must

```
make u0-key ARGS="--causal"
```

The filter has the same order and corners, but runs forward only. Does the
spike count change? By how much, and why?

<details>
<summary>Answer</summary>

Detections fall from 28,575 to 11,353, a loss of 60 %. With a gentler
2nd-order causal filter (`--causal --order 2`) the count is 17,256, a loss of
40 %.

Two things happen, both visible on the example clip:

- **Dispersion.** The causal 4th-order band-pass delays 400 Hz by 22 samples
  (1.1 ms) but 2 to 4 kHz by about 1 sample. A spike is a sum of these
  frequencies, so its components stop lining up and the trough gets shallower:
  -110 LSB against -132 LSB zero-phase, with a +69 LSB overshoot after it.
- **Steep amplitude distribution.** Half of all detected spikes are within
  1.05σ of the threshold, so a 15 to 20 % loss of depth pushes a large share
  below it.

The noise σ also rises 4 % (7.01 to 7.31), because one pass has a wider
effective band than two.

This is the result that matters most for later units. Every offline number in
this project is optimistic for an implant unless the filter is causal. U2
therefore compares feature paths with causal filters, and U7's RTL must match
the causal Python model, not this one.
</details>

### Check 3: throw away ADC bits

```
make u0-key ARGS="--drop-bits 3"     # a 7-bit ADC
make u0-key ARGS="--drop-bits 5"     # a 5-bit ADC
```

Quantisation adds noise of variance Δ²/12 for a step Δ. Predict σ after
dropping 3 bits (Δ = 8 LSB), then guess what happens at 5 bits (Δ = 32 LSB,
about 4.6σ).

<details>
<summary>Answer</summary>

**3 bits:** quantisation noise is 64/12 = 5.33 LSB², spread evenly from 0 to
9,766 Hz. The spike band keeps (7500 - 300)/9766 = 73.7 % of it, so

σ = √(7.01² + 0.737 × 5.33) = 7.28 LSB.

Measured: 7.28 LSB. Detections fall by 6.5 % (28,575 to 26,726). Three bits
cost almost nothing for threshold detection.

**5 bits:** the same formula predicts √(49.1 + 0.737 × 85.3) = 10.6 LSB. The
measurement is 7.84 LSB and 282,222 detections, ten times the baseline. The
detector has failed, and the formula failed with it.

Δ²/12 assumes the quantisation error is small, uniform and uncorrelated with
the signal. That holds when the noise spans several steps, because the noise
itself dithers the quantiser. At Δ ≈ 4.6σ the signal sits in one or two codes,
the error is the signal, and a single code flip is a 32 LSB step that looks
like a spike after filtering.

The lesson: bits can be removed until the step is about the size of the noise,
and then the cost stops being gradual. U2 sweeps this on MEArec data, where
ground truth says which detections are real.
</details>

### Check 4 (bonus): use the 300 to 1000 Hz spiking band

```
make u0-key ARGS="--spike-band 300 1000"
```

Nason et al. (2020) use only this band. If the noise were white, σ would fall
by √(700/7200) = 0.31. Does it?

<details>
<summary>Answer</summary>

σ falls only to 4.64 LSB, a factor of 0.66, not 0.31. The noise is not white:
its PSD falls as f^-1.27, so most of the spike-band noise power is already in
the bottom of the band, and removing the top removes little.

Spike depth falls by a similar factor (43.7 to 28.7 LSB), so the detected SNR
is nearly unchanged (5.38σ against 5.55σ). But detections fall from 28,575 to
8,972: narrowing the band widens each spike in time and lowers its peak, and
many slip under the threshold.

This band is not designed for a threshold detector. Nason's feature is the
average absolute value of this band (spiking-band power), which integrates
energy instead of looking for a peak, and can be sampled at 2 kHz instead of the
20 to 30 kHz that spike shapes need. Whether that trade wins is the question U2 and U3 answer.
</details>

## 6. What to read next

- Quiroga et al. (2004), sections 2.1 to 2.2: the MAD threshold used here.
- Nason et al. (2020), figures 1 to 3: why a 300 to 1000 Hz band still tracks
  single units. This is the paper U2 reproduces.
- Harrison and Charles (2003): the amplifier U1 models, and where the noise
  efficiency factor comes from.

Full citations are in [references.md](references.md).
