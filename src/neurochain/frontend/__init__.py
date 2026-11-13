"""Electrode and analog front-end models.

* :mod:`neurochain.frontend.electrode`: electrode-tissue impedance and its
  thermal noise.
* :mod:`neurochain.frontend.lna`: a capacitive-feedback low-noise amplifier.
* :mod:`neurochain.frontend.model`: electrode plus amplifier solved as one
  network: gain, bandwidth and noise.
* :mod:`neurochain.frontend.hand`: the closed-form hand calculation.
* :mod:`neurochain.frontend.spice`: the same circuit in ngspice.
* :mod:`neurochain.frontend.coupling`: DC-offset tolerance and settling.
* :mod:`neurochain.frontend.emulator`: applies the model to a recording.
"""
