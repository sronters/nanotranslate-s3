.PHONY: host test smoke research clean

host:
	cmake -S host -B host/build
	cmake --build host/build -j

test:
	PYTHONPATH=model python -m pytest -q

smoke: host
	PYTHONPATH=model python tests/smoke_e2e.py

research: host
	PYTHONPATH=model python experiments/analytic_sweep.py --out-dir research/data
	PYTHONPATH=model python experiments/smoke_quantization.py --out-dir research/data

clean:
	rm -rf host/build .pytest_cache
