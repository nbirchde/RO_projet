# Makefile for setting up, running, and cleaning the fair round-robin scheduler

.PHONY: venv install run runM sa saM sa_non_opti sa_non_optiM benchmark benchmarkM calibrate calibrateM plot plotM clean

# Setup virtual environment
venv:
	python3 -m venv venv

# Install dependencies
install: venv
	source venv/bin/activate && pip install -r Requirements.txt

# Run the Exact Solver (MILP) - Example for n=6 with time limit
run:
	source venv/bin/activate && python3 -m src.exact_model 6 0.8 1.2 10

# Run the Exact Solver (MILP) on Windows - Example for n=6 with time limit
runM:
	python.exe -m src.exact_model 6 0.8 1.2 10
	
# Run the Optimized SA Solver - Example for n=10 with time budget
sa:
	source venv/bin/activate && python3 -m src.sa_solver 10 -t 10

# Run the Optimized SA Solver on Windows - Example for n=10 with time budget
saM:
	python.exe -m src.sa_solver 10 -t 10

# Run the Non-Optimized SA Solver - Example for n=10 with time budget
sa_non_opti:
	source venv/bin/activate && python3 -m src.sa_solver_non_opti 10 10

# Run the Non-Optimized SA Solver on Windows - Example for n=10 with time budget
sa_non_optiM:
	python.exe -m src.sa_solver_non_opti 10 10

# Run the SA Benchmark script
benchmark:
	source venv/bin/activate && python3 benchmark_sa_algorithms.py

# Run the SA Benchmark script on Windows
benchmarkM:
	python.exe benchmark_sa_algorithms.py

# Run the Calibration script (generates calibration_results_nXXX_analytical_norm.csv)
calibrate:
	source venv/bin/activate && python3 -m src.run_calibration

# Run the Calibration script on Windows
calibrateM:
	python.exe -m src.run_calibration

# Run the Plotting script (generates plots from calibration results)
plot:
	source venv/bin/activate && python3 -m src.plot_calibration_results

# Run the Plotting script on Windows
plotM:
	python.exe -m src.plot_calibration_results

# Clean up generated files and directories
clean:
	rm -f sa_algorithms_comparison_results.csv
	rm -f raw_metrics_comparison.png
	rm -f sa_schedule_n*.csv
	rm -f calibration_results_n*.csv
	rm -rf calibration_plots_n*/
	find . -name '.DS_Store' -delete
	rm -f final_report_files/*.aux final_report_files/*.log final_report_files/*.out final_report_files/*.fls final_report_files/*.fdb_latexmk final_report_files/*.synctex.gz final_report_files/*.toc
