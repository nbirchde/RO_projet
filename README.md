# Fair Round-Robin Tournament Scheduling

This project explores different methods for generating fair round-robin tournament schedules, minimizing a weighted combination of fairness metrics using empirical normalization.

## Solvers

The project includes the following solvers:

### Exact MILP Solver

Uses a Mixed-Integer Linear Program (MILP) to find optimal solutions for smaller instances.

To run the exact solver:

```bash
python3 src/exact_model.py [n_players] [alpha] [beta] [time_limit]
```

Parameters:
- `n_players` (int, required): Number of players (must be even).
- `alpha` (float, optional): Weight for the Penalty Sequence objective term. Defaults to the value in `src/config.py`.
- `beta` (float, optional): Weight for the Max Deviation objective term. Defaults to the value in `src/config.py`.
- `time_limit` (int, optional): Time limit for the solver in seconds. Defaults to None (no limit).

Example:
```bash
python3 src/exact_model.py 6 0.9 1.3 60
```

### Simulated Annealing (Non-Optimized)

A basic Simulated Annealing heuristic implementation.

To run the non-optimized SA solver:

```bash
python3 src/sa_solver_non_opti.py [n_players] [iterations] [alpha] [beta]
```

Parameters:
- `n_players` (int, required): Number of players (must be even).
- `iterations` (int, optional): Number of SA iterations. Defaults to 10000.
- `alpha` (float, optional): Weight for the Penalty Sequence objective term. Defaults to the value in `src/config.py`.
- `beta` (float, optional): Weight for the Max Deviation objective term. Defaults to the value in `src/config.py`.

Example:
```bash
python3 src/sa_solver_non_opti.py 8 50000 0.9 1.3
```

### Simulated Annealing (Numba Optimized)

A Numba-optimized version of the Simulated Annealing heuristic for better performance on larger instances. This solver also stores and loads the best-found schedule for a given `n` to potentially improve results across multiple runs.

To run the Numba-optimized SA solver:

```bash
python3 src/sa_solver.py [n_players] [iterations] [alpha] [beta] [runs]
```

Parameters:
- `n_players` (int, required): Number of players (must be even).
- `iterations` (int, optional): Number of SA iterations per run. Defaults to 100000.
- `alpha` (float, optional): Weight for the Penalty Sequence objective term. Defaults to the value in `src/config.py`.
- `beta` (float, optional): Weight for the Max Deviation objective term. Defaults to the value in `src/config.py`.
- `runs` (int, optional): Number of parallel SA runs. Defaults to 1.

Other SA parameters (initial temperature, cooling rate, seed, empirical sample size, log interval) use default values defined within the script (`src/sa_solver.py`).

Example:
```bash
python3 src/sa_solver.py 10 100000 0.9 1.3 4
```

## Normalization Data

Empirical normalization factors (medians and standard deviations) and the best-found schedules for each number of players (`n`) are stored in `normalization_data.json`. This file is automatically created and updated by the solvers.

## LaTeX Report

The scientific report is written in LaTeX. To compile the report:

Navigate to the `final_report_files` directory and run:

```bash
latexmk -pdf -output-directory=final_report_files final_report_files/rapport_scientifique.tex
```

You may need to run this command multiple times (typically 2-3 times) to ensure cross-references and the table of contents are generated correctly.
