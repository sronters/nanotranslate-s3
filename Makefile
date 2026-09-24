.PHONY: host test smoke research paper clean
host:
	cmake -S host -B host/build
	cmake --build host/build -j

test:
	PYTHONPATH=model pytest -q

smoke: host
	PYTHONPATH=model python tests/smoke_e2e.py

research: host
	PYTHONPATH=model python experiments/analytic_sweep.py --out-dir research/data
	PYTHONPATH=model python experiments/smoke_quantization.py --out-dir research/data

paper: research
	cd research && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	rm -rf host/build .pytest_cache research/*.aux research/*.fdb_latexmk research/*.fls research/*.log research/*.out
