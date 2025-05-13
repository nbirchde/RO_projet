import numpy as np
import random
import json
import os
import logging
import sys

# Add the project root directory to sys.path to enable importing modules from src
# This allows running the script directly from the project root.
# This might be redundant if the script is always run from the project root,
# but it's safer for imports within the src directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

# Import necessary functions from metrics.py for raw metric calculation
from .metrics import calculate_home_strength, calculate_raw_max_deviation, calculate_raw_total_penalty_sequence

# Import schedule generator utility
from .schedule_utils import initial_schedule

log = logging.getLogger(__name__)

NORMALIZATION_DATA_FILE = 'normalization_data.json'

def load_normalization_data():
    """Loads normalization data from the JSON file."""
    if os.path.exists(NORMALIZATION_DATA_FILE):
        with open(NORMALIZATION_DATA_FILE, 'r') as f:
            try:
                data = json.load(f)
                # Ensure data structure is a dictionary
                if not isinstance(data, dict):
                    log.warning(f"Data in {NORMALIZATION_DATA_FILE} is not a dictionary. Starting with empty data.")
                    return {}
                return data
            except json.JSONDecodeError:
                log.warning(f"Could not decode JSON from {NORMALIZATION_DATA_FILE}. Starting with empty data.")
                return {}
    return {}

def save_normalization_data(data):
    """Saves normalization data to the JSON file."""
    # Ensure data is a dictionary before saving
    if not isinstance(data, dict):
        log.error("Attempted to save non-dictionary data to normalization_data.json.")
        return
    with open(NORMALIZATION_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_best_schedule(n):
    """Loads the best schedule and its normalized score for a given n."""
    data = load_normalization_data()
    n_str = str(n)
    if n_str in data and 'best_schedule' in data[n_str] and 'best_norm_score' in data[n_str]:
        log.info(f"Loading best schedule for n={n} from {NORMALIZATION_DATA_FILE}.")
        return data[n_str]['best_schedule'], data[n_str]['best_norm_score']
    log.info(f"No best schedule found for n={n} in {NORMALIZATION_DATA_FILE}.")
    return None, None

def save_best_schedule(n, schedule, norm_score):
    """Saves the best schedule and its normalized score for a given n."""
    data = load_normalization_data()
    n_str = str(n)
    if n_str not in data:
        data[n_str] = {} # Create entry if it doesn't exist

    # Preserve existing normalization factors if they exist
    current_factors = {
        'med_hs': data[n_str].get('med_hs'),
        'sigma_hs': data[n_str].get('sigma_hs'),
        'med_ps': data[n_str].get('med_ps'),
        'sigma_ps': data[n_str].get('sigma_ps'),
        'med_md': data[n_str].get('med_md'),
        'sigma_md': data[n_str].get('sigma_md')
    }

    # Convert NumPy int64 to standard Python int for JSON serialization
    serializable_schedule = []
    for round_matches in schedule:
        serializable_round = []
        for home, away in round_matches:
            serializable_round.append((int(home), int(away))) # Convert int64 to int
        serializable_schedule.append(serializable_round)

    data[n_str] = current_factors # Restore existing data
    data[n_str]['best_schedule'] = serializable_schedule # Save the converted schedule
    data[n_str]['best_norm_score'] = norm_score

    save_normalization_data(data)
    log.info(f"Saved best schedule for n={n} with score {norm_score:.4f} to {NORMALIZATION_DATA_FILE}.")


def calculate_empirical_factors(n, num_samples, schedule_generator, seed=42):
    """
    Generates a sample of schedules and computes median and std dev for HS, PS, MD.
    Requires a schedule_generator function that takes n and returns a schedule list.
    """
    random.seed(seed)
    np.random.seed(seed)

    schedules_hs = []
    schedules_ps = []
    schedules_md = []

    log.info(f"Generating {num_samples} samples for n={n} to calculate empirical normalization factors...")

    for i in range(num_samples):
        sched_list = schedule_generator(n)
        if not sched_list and n > 1:
             log.warning(f"Schedule generator returned empty schedule for n={n}, sample {i}. Skipping sample.")
             continue # Skip this sample if schedule is empty

        # compute_metrics is assumed to be available via import from metrics.py
        # It returns: raw_home_strength, penalites_sequence, max_dev
        hs = calculate_home_strength(sched_list, n)
        ps = calculate_raw_total_penalty_sequence(sched_list, n)
        md = calculate_raw_max_deviation(sched_list, n)

        schedules_hs.append(hs)
        schedules_ps.append(ps)
        schedules_md.append(md)

    if not schedules_hs: # If all samples failed or num_samples was 0
        log.warning(f"No samples collected for n={n}. Using default sigmas=1.0, medians=0.0.")
        # Return placeholder values that won't cause division by zero
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    med_hs = np.median(schedules_hs)
    sigma_hs = np.std(schedules_hs, ddof=1) if len(schedules_hs) > 1 else 1.0
    med_ps = np.median(schedules_ps)
    sigma_ps = np.std(schedules_ps, ddof=1) if len(schedules_ps) > 1 else 1.0
    med_md = np.median(schedules_md)
    sigma_md = np.std(schedules_md, ddof=1) if len(schedules_md) > 1 else 1.0

    # Handle cases where sigma might be zero or very small
    sigma_hs = max(sigma_hs, 1.0)
    sigma_ps = max(sigma_ps, 1.0)
    sigma_md = max(sigma_md, 1.0)

    log.info(f"Calculated empirical factors for n={n}: med_hs={med_hs:.2f}, sigma_hs={sigma_hs:.2f}, med_ps={med_ps:.2f}, sigma_ps={sigma_ps:.2f}, med_md={med_md:.2f}, sigma_md={sigma_md:.2f}")
    return med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md

def get_or_calculate_normalization_factors(n, schedule_generator=initial_schedule, num_samples=None, force_recalculate=False, seed=42):
    """
    Gets normalization factors for n. Loads from file if available,
    otherwise calculates, saves, and returns.
    Uses initial_schedule from schedule_utils as the default generator.
    Preserves existing best schedule data if it exists.
    """
    data = load_normalization_data()
    n_str = str(n)

    existing_data = data.get(n_str, {}) # Get existing data for this n, or empty dict

    if n_str in data and not force_recalculate and 'med_hs' in existing_data: # Check if factors exist
        log.info(f"Loading empirical factors for n={n} from {NORMALIZATION_DATA_FILE}.")
        factors = existing_data # Use existing data
        # Ensure sigmas are floats and handle potential missing keys gracefully
        med_hs = float(factors.get('med_hs', 0.0))
        sigma_hs = float(factors.get('sigma_hs', 1.0))
        med_ps = float(factors.get('med_ps', 0.0))
        sigma_ps = float(factors.get('sigma_ps', 1.0))
        med_md = float(factors.get('med_md', 0.0))
        sigma_md = float(factors.get('sigma_md', 1.0))
        # Ensure loaded sigmas are not zero
        sigma_hs = max(sigma_hs, 1.0)
        sigma_ps = max(sigma_ps, 1.0)
        sigma_md = max(sigma_md, 1.0)

        log.info(f"Loaded factors for n={n}: med_hs={med_hs:.2f}, sigma_hs={sigma_hs:.2f}, med_ps={med_ps:.2f}, sigma_ps={sigma_ps:.2f}, med_md={med_md:.2f}, sigma_md={sigma_md:.2f}")
        return med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md
    else:
        log.info(f"Normalization data for n={n} not found or recalculation forced. Calculating...")
        # Determine number of samples based on n (adaptive sampling)
        if num_samples is None:
            if n <= 50:
                effective_num_samples = 200
            elif n <= 100:
                effective_num_samples = 100
            else: # n > 100
                effective_num_samples = 50
        else:
            effective_num_samples = num_samples

        med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md = calculate_empirical_factors(
            n, effective_num_samples, schedule_generator, seed=seed
        )

        # Store the calculated data, merging with existing best schedule data if any
        data[n_str] = {
            'med_hs': med_hs,
            'sigma_hs': sigma_hs,
            'med_ps': med_ps,
            'sigma_ps': sigma_ps,
            'med_md': med_md,
            'sigma_md': sigma_md,
            # Preserve existing best schedule data if it exists
            'best_schedule': existing_data.get('best_schedule'),
            'best_norm_score': existing_data.get('best_norm_score')
        }
        save_normalization_data(data)
        log.info(f"Calculated and saved empirical factors for n={n} to {NORMALIZATION_DATA_FILE}.")

        return med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md

# This function will be used by solvers to calculate the normalized score
def calculate_normalized_score(raw_hs, raw_ps, raw_md, alpha, beta, med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md):
    """Calculates the combined normalized score using empirical factors."""
    # Ensure sigmas are not zero (should be handled by get_or_calculate_normalization_factors)
    obj_hs = (raw_hs - med_hs) / sigma_hs
    obj_ps = (raw_ps - med_ps) / sigma_ps
    obj_md = (raw_md - med_md) / sigma_md
    return obj_hs + alpha * obj_ps + beta * obj_md
