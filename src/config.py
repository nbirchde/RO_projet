#!/usr/bin/env python3
"""
Configuration file for the Round Robin project.

Stores global constants, including the chosen weights for the objective function.
"""

# Chosen weights for the normalized objective function:
# Z_norm = HomeStrength_norm + ALPHA * penseq_norm + BETA * maxdev_norm
#
# NOTE: The HomeStrength metric definition and its normalization have been updated.
# The ALPHA and BETA values below were chosen based on the *previous* HomeStrength
# definition. They will likely require re-calibration to achieve the desired balance
# with the new HomeStrength metric.
#
# Previous calibration data (now outdated for HomeStrength):
# reports/alpha_beta_calibration_n6_quick.csv
# reports/pareto_n6_quick.png

ALPHA = 0.9 # Updated to new chosen value.
BETA = 1.3  # Updated to new chosen value.

# Other configurations can be added here as needed.
