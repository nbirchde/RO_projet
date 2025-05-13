# RO-PROJET



6.  **Compiler le rapport LaTeX (si LaTeX est installé) :**
    ```bash
    # Depuis la racine du projet, compile et place tous les fichiers dans final_report_files
    latexmk -pdf -output-directory=final_report_files final_report_files/rapport_scientifique.tex
    ```

## Alpha/Beta Calibration and Pareto Plots

This project includes scripts to perform a calibration of the `alpha` and `beta` weights used in the simulated annealing solver's objective function and to visualize the resulting Pareto frontiers.

### 1. Running the Calibration

The calibration script (`run_calibration.py`) executes the simulated annealing solver (`src/sa_solver.py`) for a grid of `alpha` and `beta` values. It collects the normalized objective function components (HomeStrength, PenaltySequence, MaxDeviation) for each combination.

To run the calibration (this may take several minutes for the dense grid):
```bash
python3 run_calibration.py
```
This will generate a CSV file named `calibration_results_n6_dense.csv` (or `calibration_results_n6_new.csv` if you use the older configuration in the script) in the project root.

The script is configured by default for `n=6` players, `10000` iterations per SA run, and `4` parallel SA chains per (alpha, beta) combination. The `ALPHA_VALUES` and `BETA_VALUES` in `run_calibration.py` are set to range from 0.0 to 2.0 with a step of 0.1, resulting in 441 combinations.

### 2. Generating Pareto Plots

After the calibration data is generated, the plotting script (`plot_calibration_results.py`) can be used to visualize the results. It identifies Pareto-optimal solutions and generates:
- An interactive 3D scatter plot.
- 2D scatter plot projections for each pair of objectives.
- Static PNG versions of these plots suitable for inclusion in reports.

To generate the plots:
```bash
python3 plot_calibration_results.py
```
This script will read the `calibration_results_n6_dense.csv` file and save the plots into the `calibration_plots_dense/` directory.

### 3. Viewing the Plots

- **Interactive HTML plots:** Open the `.html` files in the `calibration_plots_dense/` directory (e.g., `calibration_plots_dense/pareto_3d_interactive.html`) in a web browser.
- **Static PNG plots:** These can be viewed directly and are suitable for embedding in documents.

The plotting script will also print a list of suggested Pareto-optimal (alpha, beta) combinations to the console, which can help in selecting appropriate weights for `src/config.py`.
