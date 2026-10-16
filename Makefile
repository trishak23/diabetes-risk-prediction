install:
	pip install -r requirements.txt

data:
	python -m src.diabetes_risk.data

train:
	python scripts/run_pipeline.py

test:
	python -m unittest discover -s tests -v

run: train

.PHONY: install data train test run
