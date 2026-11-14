"""On-implant feature extraction: spike detection and spiking-band power.

Everything here runs forward in time only, as an implant must.

* :mod:`neurochain.features.detect`: threshold and nonlinear-energy detectors.
* :mod:`neurochain.features.sbp`: spiking-band power and feature binning.
* :mod:`neurochain.features.score`: scoring detections against ground truth.
"""
