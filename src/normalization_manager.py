import numpy as np
import math
import logging
import sys
import os

# Add the project root directory to sys.path to enable importing modules from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

log = logging.getLogger(__name__)

def calculate_analytical_factors(n):
    """
    Calculates the analytical mean and standard deviation for HS, PS, and MD
    based on the provided formulas for a random Home/Away assignment.
    """
    if n < 2:
        log.warning(f"Cannot calculate analytical factors for n={n}. Must be n >= 2.")
        # Return placeholder values that won't cause division by zero
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)

    R = n - 1

    # 1. HomeStrength (HS)
    # E[HS] = n*(n-1)*(n+1)/12
    # sigma_HS = n*sqrt(n^2 - 1)/(4*sqrt(3))
    mu_HS = n * (n - 1) * (n + 1) / 12
    sigma_HS = n * math.sqrt(n**2 - 1) / (4 * math.sqrt(3))

    # 2. Penalty Sequence (PS)
    # E[PS] = n*(R-1)/2
    # sigma_PS = sqrt(n*(R-1))/2
    mu_PS = n * (R - 1) / 2
    sigma_PS = math.sqrt(n * (R - 1)) / 2 if R > 1 else 1.0 # Handle R=1 case (n=2)

    # 3. Max Deviation (MD)
    # E[MD] ~ sqrt(R)/2 * sqrt(2*ln n)
    # sigma_MD ~ sqrt(R)/2 * pi / sqrt(6*ln n)
    if n > 1: # ln(1) is undefined, MD is 0 for n=1
        mu_MD = 0.5 * math.sqrt(R) * math.sqrt(2 * math.log(n))
        sigma_MD = 0.5 * math.sqrt(R) * math.pi / math.sqrt(6 * math.log(n)) if n > 1 else 1.0 # Handle n=1 case
    else:
        mu_MD = 0.0
        sigma_MD = 1.0 # Default sigma for n=1

    # Ensure sigmas are not zero or very small
    sigma_HS = max(sigma_HS, 1e-9)
    sigma_PS = max(sigma_PS, 1e-9)
    sigma_MD = max(sigma_MD, 1e-9)

    log.info(f"Calculated analytical factors for n={n}:")
    log.info(f"  HS: Mu={mu_HS:.4f}, Sigma={sigma_HS:.4f}")
    log.info(f"  PS: Mu={mu_PS:.4f}, Sigma={sigma_PS:.4f}")
    log.info(f"  MD: Mu={mu_MD:.4f}, Sigma={sigma_MD:.4f}")

    return (mu_HS, sigma_HS), (mu_PS, sigma_PS), (mu_MD, sigma_MD)

# This function will be used by solvers to calculate the normalized score
# It now uses analytical factors directly
def calculate_normalized_score(raw_hs, raw_ps, raw_md, alpha, beta, n):
    """
    Calculates the combined normalized score using analytical factors.
    Normalization: z = (raw - mu) / sigma
    Objective: Z = z_HS + alpha * z_PS + beta * z_MD
    """
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    # Calculate z-scores
    obj_hs = (raw_hs - mu_hs) / sigma_hs
    obj_ps = (raw_ps - mu_ps) / sigma_ps
    obj_md = (raw_md - mu_md) / sigma_md

    # Calculate combined objective score
    total_normalized_score = obj_hs + alpha * obj_ps + beta * obj_md

    return total_normalized_score, obj_hs, obj_ps, obj_md

# Remove functions related to empirical normalization and file handling:
# load_normalization_data
# save_normalization_data
# get_best_schedule
# save_best_schedule
# calculate_empirical_factors
# get_or_calculate_normalization_factors
