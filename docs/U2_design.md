# U2 design: ADC and features

**Done-criteria:** detection F1 on MEArec across noise levels; SBP against
threshold crossings reproduced on the Nason data.
**Status:** both met. `make mearec` generates the ground-truth recording and
`make u2` reproduces everything.

Every number below comes from `results/u2_metrics.json` or from
`scripts/u2_key_number.py`.

## 1. What U2 has to answer

U0 and U1 could only count threshold crossings. Nothing said whether a crossing
was a real spike. U2 adds ground truth, and then asks the questions the power
budget depends on:

1. How good is spike detection that an implant could actually run, at the noise
   level of a real brain?
2. How few ADC bits and how low a sample rate can it survive?
3. Is spiking-band power (SBP), which needs a tenth of the sample rate, as good
   a feature as counting spikes?
4. What breaks when the arithmetic becomes integer-only, as it must in hardware?

## 2. The chain under test

```
MEArec spikes + neural background
   -> front-end emulator (U1: gain, band, electrode and amplifier noise)
   -> ADC (sample rate, bits, input range)
   -> causal high-pass at 300 Hz
   -> detector (threshold or NEO)  -> scored against the true spike times
   or |x| averaged                 -> spiking-band power
```

**MEArec** (Buccino and Einevoll, 2021) simulates extracellular recordings from
detailed cortical neuron models, so every spike's time and source is known. The
recording here is 32 channels, 16 units, 60 s at 30 kHz.

The generation script builds it twice from the same spike trains: once clean,
once with MEArec's far-neuron background (300 distant neurons too small to
detect). The difference is the background alone, which the loader rescales. So
"noise level" in this unit means **the spike-band RMS of the neural
background**, in the same units as the 13.2 µV measured on the Nason channel in
U1. The front end's own 3.4 µV is added on top by the emulator.

## 3. Decisions

### 3.1 Scoring: who counts as a spike the detector should find

A detector cannot be asked to find every spike of every simulated neuron; most
are far too small on any given channel. The rules:

- A **target** spike comes from a unit whose trough is at least 50 µV deep on
  that channel (MEArec's own default for a detectable unit).
- A detection that lands on a spike of a smaller unit, down to 10 µV, is **set
  aside**: it is a real spike, so it is not a false positive, but it was not
  required, so it is not a true positive.
- Anything else is a **false positive**.
- A match needs to be within 1 ms.
- Thresholds are set on the first 10 s of each channel and scored on the other
  50 s, so no score uses data its threshold has seen.

F1 pools counts over all 32 channels. The interval is a bootstrap over
channels, because channels, not spikes, are the independent units.

*Course link, MATH2099:* precision, recall, resampling for a confidence
interval.

### 3.2 Two detectors

| | Score | Threshold rule | Cost in hardware |
|---|---|---|---|
| Threshold | `-x` | 4.5 × noise σ (MAD) | one comparator |
| NEO | `x[n]² − x[n−1]·x[n+1]`, smoothed over 0.2 ms | 8 × its mean | two multipliers, an adder, a short sum |

The nonlinear energy operator (Mukhopadhyay and Ray, 1998) is large when the
signal is both big and fast. For a sinusoid it equals `A²·sin²(ω)`: amplitude
squared times, roughly, frequency squared. A spike is big and fast; slow noise
is not.

Both run on a causal 2nd-order high-pass at 300 Hz. The low-pass side is the
amplifier's own roll-off, so no digital low-pass is needed.

### 3.3 ADC: specified at the electrode

The converter is described by sample rate, bits, and input range **referred to
the electrode**, with the amplifier chain's gain folded into the range. The
baseline is 30 kS/s, 10 bits, ±1 mV, a step of 1.95 µV. The range is what U0
implies for the Neuralink ADC: 146 times the spike-band noise, at an in-vivo
noise of 13 µV.

Sampling keeps every k-th sample with **no digital filter in front**, because
that is what an ADC does. Whatever the amplifier's single pole lets through
above the new Nyquist frequency folds back. The emulator also folds in the
amplifier noise from above 15 kHz, which U1 had dropped.

*Course link, ELEC3104:* sampling theorem, aliasing, quantisation noise.

### 3.4 Spiking-band power, two ways

- **As in the paper:** causal 2nd-order band-pass, 300 Hz to 1 kHz, on the
  30 kS/s signal; keep every 15th sample; absolute value. Applied to the
  broadband stored with the paper's data, this reproduces the paper's stored
  SBP to within 10⁻¹¹.
- **As an implant would:** narrow the amplifier itself to 300 Hz to 1 kHz and
  sample at 2 kS/s. The ADC then runs at a fifteenth of the rate, which is the
  point.

### 3.5 Fixed point: decisions the RTL will inherit

| Choice | 30 kS/s threshold path | 2 kS/s SBP path |
|---|---|---|
| Coefficient fractional bits | 14 | 8 |
| Guard bits | 6 | 2 |
| Rounding | round half up | round half up |
| Accumulator width needed | 29 bits | 18 bits |

Three things were learned the hard way; section 5 has the numbers.

- **Truncation is biased, and feedback multiplies the bias.** Dropping the low
  bits of a product rounds toward minus infinity: an average error of half a
  step, always the same sign. Inside a feedback loop that constant error is
  amplified by `1/(1 + a1 + a2)`. For a 300 Hz corner at 30 kS/s that gain is
  264.
- **Rounding removes the bias but leaves a deadband.** With rounding, a silent
  input can leave the output resting up to 132 internal steps from zero. Six
  guard bits shrink that to 2 codes, and the noise in any real signal dithers
  it away (measured offset: 0.008 code).
- **The noise estimate needs sub-code resolution.** The median of integer data
  lands on a whole number. With a noise of about 7 codes that is a 10 % error
  in σ, and so in the threshold. `mad_sigma_integer` interpolates the median
  inside its bin, which brought fixed-point F1 from 0.611 back to 0.632.

The high-pass numerator is built as `b0·(1 − z⁻¹)²`, so coefficient rounding can
never let DC leak through.

*Course link, ELEC2141:* two's-complement arithmetic, shifts, word growth.
*ELEC3104:* IIR structures, coefficient quantisation, limit cycles.

## 4. What the numbers say

**Detection at in-vivo noise (13 µV), 30 kS/s, 10 bits:**

| Detector | F1 | Precision | Recall |
|---|---|---|---|
| Threshold, causal | 0.63 (0.58 to 0.68) | 0.72 | 0.57 |
| NEO, causal | 0.64 (0.58 to 0.69) | 0.88 | 0.50 |
| Threshold, zero-phase | 0.66 (0.61 to 0.71) | 0.64 | 0.69 |

1. **F1 is modest, and noise sets it.** 0.89 at 5 µV, 0.63 at 13 µV, 0.38 at
   20 µV. At 13 µV the threshold sits at about 60 µV, above the smallest target
   units, so recall cannot be high. This is the neural background, not the
   electronics.
2. **NEO does not beat a plain threshold on F1.** It trades recall for
   precision. Sweeping each detector's setting moves along nearly the same
   curve.
3. **The causal penalty is small when the filter is gentle.** U0 saw a 60 % loss
   of crossings with a causal 4th-order band-pass. Here, with a 2nd-order
   high-pass and the amplifier's own roll-off, causal costs 0.03 of F1. The
   penalty belonged to that filter, not to causality.
4. **Six bits are enough; five are not.** F1 is 0.63 at 12, 10 and 8 bits, 0.61
   at 6, and 0.12 at 5. At 5 bits the step (62 µV) is nearly five times the
   noise: most samples are code zero, the noise estimate collapses and the
   threshold with it.
5. **Sample rate degrades gracefully.** 0.63 at 15 kS/s, 0.61 at 10 kS/s, 0.53
   at 5 kS/s.
6. **SBP reproduces, and matches crossings for decoding.** On the Nason online
   data the same linear readout gives R² = 0.47 from SBP and 0.46 from
   crossings for finger velocity; together 0.52, so they are not redundant. In
   vivo, SBP follows the largest unit (r = 0.75) where crossings do not
   (0.19), which is the paper's claim that SBP is dominated by single units.
7. **SBP is not uniformly better.** On MEArec at 13 µV it tracks the largest
   unit at r = 0.50 against 0.61 for crossings. It wins at 5 µV and below and
   loses from 10 µV up, because averaging |x| also averages the noise. U3 has to weigh that
   against a fifteenth of the ADC rate.
8. **Integer-only arithmetic costs nothing once done properly**: F1 0.632
   against 0.634, SBP correlation 0.999.

### Limitations

- **MEArec is a simulation.** Its background has more than half its power below
  300 Hz and no LFP as large as a real one. The Nason broadband has an RMS of
  972 µV, which would clip a ±1 mV converter. The input range and the analog
  high-pass corner therefore still have to be chosen on real data; U3 does.
- **Constant firing rates.** MEArec units fire as Poisson processes at fixed
  rates, so "tracking a unit" means following its random fluctuations, not a
  behaviourally driven rate.
- **NEO was not retuned per setting.** Its conventional factor of 8 is tuned at
  30 kS/s. Its score depends on the sample rate and the step size, which is why
  its curves in the ADC sweeps are irregular. A fair comparison at each rate
  would retune it.
- **The target threshold of 50 µV is a choice.** F1 moves with it. Comparisons
  within this unit share the choice, so they hold; the absolute value is not a
  universal number.
- **Threshold F1 is lower with no background noise than with 5 µV.** With only
  front-end noise the threshold drops to about 15 µV and starts counting units
  below the 10 µV set-aside limit and filter ringing. A threshold relative to
  noise is not automatically better when noise falls.
- **The readout on the Nason data is linear.** U4 replaces it with a Kalman
  filter and a GRU.

## 5. Prediction checks

Read the question, write down what you expect, run the command, then open the
answer. Baseline at 13 µV, 30 kS/s, 10 bits: threshold F1 0.633 (precision
0.72, recall 0.57), NEO F1 0.637 (precision 0.89, recall 0.50).

### Check 1: cut the ADC from 10 bits to 6, then to 5

```
make u2-key ARGS="--bits 6"
make u2-key ARGS="--bits 5"
```

The step goes from 1.95 µV to 31 µV, then 62 µV. The noise is about 13 µV. What
happens to each detector?

<details>
<summary>Answer</summary>

**6 bits (step 31 µV, 2.4σ):** threshold F1 falls only from 0.633 to 0.611.
NEO falls from 0.637 to 0.411: its precision rises to 0.99 but its recall
halves, to 0.26.

**5 bits (step 62 µV, 4.7σ):** both collapse. Threshold F1 is 0.123, with
recall 0.95 and precision 0.07: it fires on almost everything.

The threshold detector needs only one thing from the ADC: to tell "below the
threshold" from "not". Even a 31 µV step leaves a 60 µV threshold two codes
away, so it survives. NEO multiplies samples together, which squares the
quantisation error, so it suffers first.

At 5 bits the failure is not in the detector but in its calibration. Most
samples are code zero, the median absolute deviation is zero or one code, and
the threshold lands on top of the noise.

The raw data rate goes from 300 to 180 kbit/s for 0.02 of F1. This is the
cheapest saving in the whole chain.
</details>

### Check 2: sample at a third of the rate

```
make u2-key ARGS="--fs 10000"
```

Spikes are about 0.3 ms wide: three samples at 10 kS/s. Nothing but the
amplifier's single pole at 7.5 kHz stands between the signal and the sampler.
How much F1 is lost?

<details>
<summary>Answer</summary>

Threshold F1 falls from 0.633 to 0.610. Precision actually rises (0.72 to
0.79) while recall falls (0.57 to 0.50): with three samples per spike the
sampler often misses the trough, so small spikes drop below the threshold.

NEO falls to 0.511, and in the opposite way: precision collapses to 0.38 while
recall rises to 0.79. Its score scales with `sin²(ω)`, so at a third of the
rate the same signal produces a different score distribution and the factor of
8, tuned at 30 kS/s, is wrong.

The lesson: a threshold detector tolerates undersampling far better than its
"20 to 30 kS/s" convention suggests. The convention exists for spike sorting,
which needs the waveform's shape. Detection only needs to see that something
crossed.
</details>

### Check 3: raise the neural background from 13 µV to 20 µV

```
make u2-key ARGS="--noise 20"
```

About 50 % more noise. How much F1?

<details>
<summary>Answer</summary>

Threshold F1 falls from 0.633 to 0.382; NEO from 0.637 to 0.390. Recall nearly
halves (0.57 to 0.30) and precision falls as well (0.72 to 0.52).

The threshold is tied to the noise, so it rises from about 60 µV to about
90 µV, above most of the target units. Half of what was detectable is now
below the threshold by construction.

Compare this with check 1. Halving the ADC's resolution four times over cost
0.02. A 50 % rise in neural background cost 0.25. Detection is limited by the
brain, not by the converter, which is the same conclusion U1 reached about the
amplifier.
</details>

### Check 4 (bonus): truncate instead of rounding

No command is needed; the numbers are in `results/u2_metrics.json` under
`fixed_point`, and `tests/test_fixedpoint.py` reproduces them.

A fixed-point filter drops the low bits of each product. Dropping them
(truncation) costs nothing; adding half a step first (rounding) costs one
adder. How much does skipping the adder matter for a 300 Hz high-pass at
30 kS/s? And at 2 kS/s?

<details>
<summary>Answer</summary>

At 30 kS/s with no guard bits, truncation shifts the output by 131 codes, a
quarter of the ADC's negative range. At 2 kS/s the same filter is off by 0.95
of a code.

Truncation is wrong by half a step on average, always downward. That constant
enters the feedback loop and is amplified by `1/(1 + a1 + a2)`: 264 at
30 kS/s, 1.9 at 2 kS/s. Half a step times 264 is 132.

The loop gain is large when the corner is far below the sample rate, because
the poles sit close to `z = 1`. Lower the sample rate and the same corner is
easy. This is a second, separate argument for the 2 kS/s SBP path: its filter
needs an 18-bit accumulator where the 30 kS/s one needs 29.
</details>

## 6. What to read next

- Nason et al. (2020), figures 1 to 4: the claims reproduced here, with the
  authors' own decoders.
- Mukhopadhyay and Ray (1998), sections II and III: why the energy operator
  favours spikes.
- Buccino and Einevoll (2021), section 2: how the recording was simulated and
  what the far-neuron background is.

Full citations are in [references.md](references.md).
