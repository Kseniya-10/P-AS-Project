.PHONY: install run load transform status dashboard test

install:
	pip install -r requirements.txt
run:
	python -m pipeline run
load:
	python -m pipeline run --stage load
transform:
	python -m pipeline run --stage transform
status:
	python -m pipeline status
dashboard:
	python -m streamlit run app.py
test:
	pytest -q
