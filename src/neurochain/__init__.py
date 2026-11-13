"""NeuroChain: a neural sensing chain modelled end to end under a power budget.

Sub-packages:

* :mod:`neurochain.data`: dataset registry, downloader and loaders.
* :mod:`neurochain.dsp`: band-splitting, spectra and spike detection.
* :mod:`neurochain.analysis`: per-unit analyses that turn recordings into the
  numbers and arrays behind each figure.
"""

from neurochain.recording import Recording

__version__ = "0.1.0"

__all__ = ["Recording", "__version__"]
