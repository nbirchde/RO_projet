# Makefile for setting up and running the fair round-robin scheduler

.PHONY: venv install run validate

venv:
	python3 -m venv venv

install: venv
	source venv/bin/activate && pip install -r Requirements.txt

run:
	source venv/bin/activate && python3 src/exact_model.py

runM:
	python.exe src/exact_model.py


validate:
	source venv/bin/activate && python3 src/validate_small_n.py

validateM:
	python.exe src/validate_small_n.py	

sa:
	source venv/bin/activate && python3 src/sa_solver.py

saM:
	python.exe src/sa_solver.py

benchmark:
	source venv/bin/activate && python3 src/benchmark_sa.py

benchmarkM:
	python.exe src/benchmark_sa.py
