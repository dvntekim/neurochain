# U1 design: electrode and front-end model

**Done-criterion:** simulated noise within 10 % of the hand calculation.
**Status:** met. The hand calculation, the analytic model and ngspice agree
within 0.3 % for three electrode models. `make u1` reproduces everything.

Every number below comes from `results/u1_metrics.json` or from
`scripts/u1_key_number.py`. U1 is the first unit that works in real microvolts.

## 1. What U1 has to answer

1. How much noise does the front end add before the signal is ever digitised,
   and which part of the front end adds it?
2. How does that noise trade against power? This is the link from circuit to the
   project's research question.
3. What does the front end do about an electrode offset thousands of times
   larger than the signal?

## 2. The circuit

```
            electrode                         amplifier
                                         +----[ C2 ]----+
 tissue                                  +----[ R2 ]----+
 signal --[ Rs ]--+--[ Rct ]--+--[ C1 ]--+---[ -gm ]----+--- out
                  +--[ CPE ]--+          |              |
                                       [ Cin ]        [ CL ]
                                         |              |
                                        gnd            gnd
```

**Electrode (Randles circuit).** `Rs` is the access resistance through
electrolyte and tissue. After it the interface: a charge-transfer resistance
`Rct` in parallel with the double layer. The values are the mean fit to 96
impedance spectra of Blackrock platinum Utah arrays in saline (Straka et al.,
2018, Table 1): `Rs` = 7.3 kΩ, `Rct` = 4.6 GΩ, and a double layer that is **not
a capacitor** but a constant-phase element (CPE), `Z = 1/(Q(jω)^n)` with
`Q` = 0.72 × 10⁻⁸ S·sⁿ and `n` = 0.74. Section 3.1 is about why that matters.

**Amplifier (Harrison and Charles, 2003).** The signal enters through a
capacitor `C1`. A small capacitor `C2` closes the feedback loop around a
transconductance amplifier (OTA), so the gain is `C1/C2` = 20 pF / 200 fF = 100,
exactly as an inverting op-amp stage has gain `Z2/Z1`. A huge resistance `R2`
across `C2` gives the OTA a DC operating point and sets the high-pass corner
`1/(2πR2C2)`. The OTA's `gm` and load `CL` set the low-pass corner.

Baseline values: 10 µA from 1.8 V (18 µW), NEF 4.0, pass band 1 Hz to 7.5 kHz,
at 37 °C.

## 3. Decisions

### 3.1 Double layer: constant-phase element, not an ideal capacitor

A single impedance measurement at 1 kHz cannot tell the two apart. Both
electrodes below are 218 kΩ at 1 kHz.

| | Measured fit (CPE, n = 0.74) | Textbook Randles (ideal capacitor) |
|---|---|---|
| Phase at 1 kHz | -65° | -88° |
| Real part of Z at 1 kHz | 93 kΩ | 7.3 kΩ |
| Thermal noise, 300 Hz to 7.5 kHz | 2.51 µV | 0.95 µV |

Thermal noise comes only from the real part of an impedance (Nyquist:
`4kT·Re{Z}`). An ideal capacitor has none, so the textbook circuit leaves only
`Rs` to make noise. A real double layer is lossy, and its loss is 12 times `Rs`
at 1 kHz. Choosing the textbook circuit would have understated electrode noise
by a factor of 2.6.

SPICE has no fractional element, so the CPE is built from 21 series RC branches
in parallel plus two tail elements. The recipe comes from an exact integral for
`s^n` (see `cpe_ladder`), and matches the CPE to about 0.1 % from 1 Hz to
100 kHz. Because the ladder is made of resistors, SPICE gives it the right
thermal noise without being told.

*Course link, ELEC2133:* impedance of RC networks, phasors, and `4kTR` noise.

### 3.2 Amplifier: behavioural, parametrised by the NEF

| Option | For | Against | Verdict |
|---|---|---|---|
| Transistor-level OTA | Shows where the noise physically comes from | No process models here; generic MOSFET models give noise numbers that mean nothing | Rejected |
| Ideal op-amp with a noise source | Simplest | No bandwidth, no link between noise and current | Rejected |
| `gm` + output resistance + one noise source, set by NEF and current | Right topology, right poles, and noise tied to published silicon | Cannot predict a new transistor design | **Chosen** |

The noise efficiency factor (Steyaert and Sansen, 1987) is

`NEF = V_ni · √( 2·I_tot / (π·U_T·4kT·BW) )`

For an amplifier with a first-order roll-off at `BW`, total noise is the white
density squared times `π/2·BW`. Substituting and rearranging gives the one
formula this unit leans on:

**`e_n = NEF · √(4kT · U_T / I_tot)`**

Noise density depends on the supply current and the NEF, and nothing else.
That is the power-noise trade-off in one line, and it is what U3 will sweep.

**Why NEF cannot be 1 here.** A transistor in weak inversion has
`gm = κ·I_D/U_T` and current noise `2q·I_D`. Referred to its gate that is
`2kT·U_T/(κ²·I_D)`. A differential pair has two such transistors. If each
carries half the supply current, `NEF = √2/κ` = 2.02 for κ = 0.7. In Harrison's
current-mirror OTA each input transistor carries a quarter of it, so the floor
is `2/κ` = 2.86. His measured amplifier scores 4.0: the remaining transistors
add about as much noise again.

**Check against the paper.** The abstract gives 2.2 µV rms, 7.2 kHz and 80 µW.
With a 5 V supply that is 16 µA, and the formula returns NEF = 4.00. Running
this model with those values at room temperature returns 2.20 µV. That confirms
the NEF formula and the model's noise bandwidth agree with each other. It does
not reproduce his transistor design, and the 5 V supply is from my memory of
the paper, which I could not re-open this session.

*Course link, ELEC2133:* `gm`, feedback gain as an impedance ratio, noise gain,
input-referred noise.

### 3.3 Three ways of quoting "the noise", and which one U1 uses

The same amplifier gives three different numbers. Mixing them up is the most
common mistake in comparing papers.

| Convention | Baseline value | Used for |
|---|---|---|
| All frequencies, output noise ÷ midband gain | 2.94 µV (amplifier only) | The NEF formula |
| Integrated 300 Hz to 7.5 kHz, referred to the tissue signal | 2.31 µV amplifier, 3.41 µV with electrode | **U1's key number**; SPICE calls it `inoise` |
| MAD σ after the amplifier's roll-off and the digital spike filter | 3.00 µV with electrode | Comparing with a real recording |

The second divides the output noise by the gain at each frequency, so it is the
noise as the tissue signal "sees" it. The third is lower because the roll-off
attenuates noise near 7.5 kHz; it attenuates signal there by the same amount.

### 3.4 Coupling: AC, with a 1 Hz corner

An electrode sits tens to hundreds of millivolts from the reference, and
drifts. The signal is microvolts.

| Option | Offset it survives | Cost |
|---|---|---|
| **AC-coupled, gain 96 (chosen)** | Any: `C1` blocks DC | Loses signals below the corner; slow recovery after a disturbance; 20 pF capacitors take area |
| DC-coupled, same gain | 8.4 mV | Rails on any real electrode |
| DC-coupled, gain 2.5 | 300 mV | ADC must resolve microvolts on a 640 mV range: 17.5 bits against 12.3 |

DC coupling is not wrong. It keeps slow signals, and it is what a 24-bit
delta-sigma front end such as the ADS1299 does (the U8 hardware). It moves the
cost from the amplifier to the ADC: 5.3 more bits. For a spike-band implant
that is the wrong trade, so the baseline is AC-coupled.

The corner is a second trade. Lower keeps more LFP but recovers more slowly:
after a 50 mV offset step the output stays on the rail 0.29 s with a 1 Hz
corner and 2.90 s with 0.1 Hz. Section 5, check 3, measures this. Stimulation
artifacts make it the central problem of U5.

*Course link, ELEC2133:* RC high-pass, time constants, output swing.

### 3.5 Chopper stabilisation: not used

A chopper modulates the signal above the amplifier's 1/f noise corner before
amplifying and demodulates after. It is the standard fix for 1/f noise and
offset in the LFP band. The spike band starts at 300 Hz, above a typical 1/f
corner, so a chopper would buy little here and costs input impedance and ripple.
It becomes relevant if a later unit wants LFP features.

### 3.6 Emulator: causal, seeded, noise shaped by the network

The emulator applies the model to a recording: high-pass and low-pass at the
network's corners, gain, noise, rails. Three choices:

- **Filters run forward only**, as the circuit does. U0 showed a zero-phase
  filter overstates detection by more than a factor of two.
- **Noise is drawn from the network's exact output spectrum**, so electrode,
  OTA and feedback-resistor noise all carry the shaping the circuit gives them.
- **Recordings must be in microvolts.** The Neuralink data stays in ADC counts
  and cannot go through the emulator: a noise model in volts cannot be applied
  to a signal with no voltage scale.

*Course link, ELEC3104:* bilinear transform of a first-order filter; shaping
white noise to a target PSD.

## 4. The hand calculation

Constants at 37 °C: `4kT` = 1.713 × 10⁻²⁰ J, `U_T = kT/q` = 26.73 mV. Band:
300 Hz to 7.5 kHz, so Δf = 7200 Hz.

**Step 1: electrode impedance at 1 kHz.** ω = 6283 rad/s, so
ω^0.74 = 646.6 and `|Z_CPE| = 1/(Q·ω^n)` = 1/(0.72 × 10⁻⁸ × 646.6) = 214.8 kΩ
at -66.6°. Adding `Rs` gives 218 kΩ at -65°, inside the 100 kΩ to 1 MΩ range
expected of a Utah electrode.

**Step 2: electrode thermal noise.** Integrate `Re{Z}` over the band, ignoring
`Rct` (it is in gigaohms).

- Access resistance: `Rs·Δf` = 7300 × 7200 = 5.26 × 10⁷ Ω·Hz.
- Double layer: `Re{Z_CPE} = cos(nπ/2) / (Q·(2πf)^n)`, a power law in `f`, so

  `∫ = cos(nπ/2) / (Q·(2π)^n) × (f2^(1-n) − f1^(1-n)) / (1−n)`

  = 0.3971 / (0.72 × 10⁻⁸ × 3.896) × (10.175 − 4.406) / 0.26
  = 1.416 × 10⁷ × 22.19 = 3.14 × 10⁸ Ω·Hz.

The double layer contributes six times what `Rs` does. Sum: 3.67 × 10⁸ Ω·Hz,
the same as a 51 kΩ resistor across the band.

`V_electrode = √(4kT × 3.67 × 10⁸)` = **2.51 µV rms**.

**Step 3: amplifier noise.**
`e_n = NEF·√(4kT·U_T/I_tot)` = 4 × √(1.713 × 10⁻²⁰ × 0.02673 / 10⁻⁵)
= 27.1 nV/√Hz.

In the band: 27.1 nV × √7200 = **2.30 µV rms**. Over all frequencies, for the
NEF: 27.1 nV × √(π/2 × 7500) = 2.94 µV rms. Power: 10 µA × 1.8 V = 18 µW.

**Step 4: total.** Independent sources add as variances (MATH2099):

`√(2.51² + 2.30²)` = **3.40 µV rms**. This is the target.

**Step 5: against simulation.**

| | Electrode | Amplifier | Total |
|---|---|---|---|
| Hand | 2.506 | 2.296 | 3.399 µV |
| Analytic network model | 2.506 | 2.305 | 3.405 µV |
| ngspice | 2.503 | 2.300 | 3.399 µV |

Largest disagreement: 0.18 % (0.27 % across all three electrode models). The
hand calculation ignored two things the simulations include: the electrode
loads the amplifier input, which raises the amplifier's noise as seen from the
tissue by 0.4 %; and `R2` adds 0.07 µV.

**Step 6: sanity check against a real electrode.** The Nason in-vivo channel is
in microvolts. Its spike-band MAD σ, measured with U0's code, is 13.23 µV. The
model's floor through the same filter is 3.00 µV. The real channel is 4.4 times
noisier.

That is not a failed check. The model contains only the electrode's thermal
noise and the amplifier. A real recording also contains the activity of
thousands of distant neurons, plus the noise of the system that recorded it.
Lempka et al. (2011) model exactly this and get 6.2 µV thermal plus 10.2 µV
biological, 12 µV in total, for their electrodes. A measured 13.2 µV sits where
that predicts.

## 5. What the numbers say

1. **Front-end noise floor: 3.41 µV rms** in the spike band, 2.51 µV from the
   electrode and 2.31 µV from an 18 µW amplifier.
2. **The amplifier is over-designed for a real brain.** Passing the Nason
   channel through the emulated front end raises σ from 12.84 to 13.26 µV
   (predicted 13.19). Cutting the bias current to a quarter, 4.5 µW, raises it
   only to 13.81 µV: 4 % more noise for 75 % less power. Neural background
   dominates, so most of an 18 µW amplifier's quietness is wasted. U3 turns
   this into the Pareto curve.
3. **Offset handling is a choice between amplifier complexity and ADC bits**,
   and the high-pass corner sets how long an artifact blinds the channel.

### Limitations

- **No 1/f noise in the amplifier.** Real amplifiers have it. For a 100 Hz
  corner it would add about 4 % to spike-band noise power, by a hand estimate
  (`f_c·ln(f2/f1)/Δf`). It matters much more below 300 Hz, so the model's
  LFP-band noise is too low.
- **Rails are a hard clip.** A real amplifier that saturates also takes time to
  leave saturation. U5 models that.
- **Half-circuit.** The reference electrode, common-mode rejection and mains
  pickup are not modelled.
- **Electrode values are for saline, and the implanted set is from rat sciatic
  nerve**, not cortex. Encapsulation in cortex adds series resistance; Lempka's
  larger thermal noise reflects that.
- **The emulator adds noise to a recording that already has some.** On real
  data it can only make a front end worse than the one that recorded it. On
  synthetic data (U2) the input is clean and the model applies exactly.
- **No aliasing.** A single-pole roll-off leaves noise above the Nyquist
  frequency that a sampler folds back. That belongs to the ADC model in U2.
- **Assumed values:** `Cin` = 2 pF, OTA DC gain 10⁴, κ = 0.7, 0.1 V of headroom.

## 6. Prediction checks

Read the question, write down what you expect and roughly how much, run the
command, then open the answer. Baseline: |Z| = 217.8 kΩ, total noise 3.405 µV,
amplifier density 27.06 nV/√Hz, pass band 0.98 Hz to 7.14 kHz, NEF 4.00, 18 µW.

### Check 1: double the double layer

```
make u1-key ARGS="--cdl-scale 2"
```

Doubling `Q` is what doubling the electrode's area does. What happens to |Z| at
1 kHz, and to the electrode's noise?

<details>
<summary>Answer</summary>

|Z| falls from 217.8 to 110.5 kΩ: almost half, but not quite, because `Rs` does
not change. The phase barely moves (-64.8° to -63.1°), because a CPE's phase
depends on `n`, not `Q`.

Electrode noise falls from 2.506 to 1.895 µV, a factor of 0.76, not the 0.71 a
pure halving of noise power would give. In the hand calculation the interface
term halves (3.14 to 1.57 × 10⁸ Ω·Hz) and the `Rs` term stays at
0.53 × 10⁸, so the sum falls by 0.57 and its square root by 0.76.

Total noise falls 12 %, to 2.981 µV. The gain rises from 95.6 to 96.8 and the
bandwidth from 7144 to 7304 Hz, because a lower-impedance electrode loads the
input capacitor less.

The lesson: a larger electrode is quieter and easier to drive. It also averages
over more neurons, which is why electrode size is its own design problem.
</details>

### Check 2: halve the amplifier's bias current

```
make u1-key ARGS="--bias-scale 0.5"
make u1-key ARGS="--bias-scale 0.5 --keep-bandwidth"
```

Predict the noise density, the total noise and the NEF. Then decide whether
your answer depends on what happens to the load capacitor.

<details>
<summary>Answer</summary>

It depends entirely on the load capacitor.

**Load unchanged.** `gm` halves, so the density rises by √2 (27.06 to
38.27 nV/√Hz) and the bandwidth halves (7144 to 3620 Hz). The two cancel: total
noise over all frequencies stays at 2.937 µV. This is the amplifier version of
kT/C noise: total noise is set by the capacitor, not by `gm`. You have saved
half the power and lost the top half of the spike band.

**Load halved to keep 7.5 kHz.** Density still rises by √2, and now nothing
cancels: all-frequency noise rises from 2.937 to 4.154 µV, spike-band amplifier
noise from 2.305 to 3.259 µV, and total front-end noise from 3.405 to 4.111 µV
(21 % more, for half the power).

**NEF is 4.00 in both cases.** It is a property of the topology, not of the
bias point: it measures how well the amplifier uses whatever current it is
given. Halving the current does not make an amplifier less efficient, only
noisier.

The lesson: "noise" and "noise efficiency" are different axes. Moving along the
current axis trades power for noise at a fixed NEF. Only a better circuit moves
the NEF.
</details>

### Check 3: lower the high-pass corner from 1 Hz to 0.1 Hz

```
make u1-key ARGS="--f-low 0.1"
```

Does the offset tolerance change? Does the settling time? Does the spike-band
noise?

<details>
<summary>Answer</summary>

**Offset tolerance: unchanged.** `C1` blocks DC whatever `R2` is. The corner
decides how fast a change in offset is forgotten, not how large an offset can
be.

**Settling: ten times slower.** The time constant goes from 0.163 s to 1.62 s.
After a 50 mV step the output stays on the rail for 2.90 s instead of 0.29 s,
and the step takes 15.6 s instead of 1.56 s to decay to the noise floor. The
emulation and the formula `τ·ln(G·ΔV/V_swing)` agree to three figures.

**Spike-band noise: unchanged** (3.405 to 3.404 µV).

The lesson: the corner trades slow-signal content against recovery time. A
0.1 Hz corner is fine for a quiet recording and expensive next to a stimulator.
</details>

### Check 4 (bonus): raise the corner to 300 Hz to throw away the LFP

```
make u1-key ARGS="--f-low 300"
```

U0 found 83 % of the signal power is below 300 Hz. Filtering it out in the
amplifier would save ADC range. Is it free?

<details>
<summary>Answer</summary>

No. Amplifier noise in the spike band rises from 2.305 to 2.584 µV (12 %), and
total noise from 3.405 to 3.600 µV.

The feedback resistance `R2` is a resistor, so it has thermal noise, and `C2`
filters that noise to below the high-pass corner. Its total is `kT/C2`: 146 µV
at the output, 1.46 µV referred to the input, whatever `R2` is. At a 1 Hz
corner all of that sits near 1 Hz and only 0.07 µV reaches the spike band. Raise
the corner to 300 Hz and it moves into the band.

The lesson: in a capacitive-feedback amplifier the high-pass corner and the
noise are linked through `kT/C2`. A larger `C2` would fix it, and would need a
larger `C1` for the same gain, which costs area. This is why such amplifiers
keep the corner low and remove LFP later, digitally.
</details>

## 7. What to read next

- Harrison and Charles (2003), sections II and III: the amplifier modelled here
  and the NEF derivation for a real OTA.
- Straka et al. (2018), section 2.5 and Table 1: where the electrode values come
  from, and how they change after implantation (the U6 topic).
- Lempka et al. (2011), the noise sections: why a real recording is noisier than
  the front end alone.

Full citations are in [references.md](references.md).
