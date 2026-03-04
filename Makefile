.PHONY: train eval latex paper all clean install

install:
	pip install -e ".[dev]"

train:
	python -m hchealth.train --config configs/clinical_demo.yaml

eval:
	python -m hchealth.evaluate --config configs/clinical_demo.yaml

latex:
	python -m hchealth.to_latex --config configs/clinical_demo.yaml

paper:
	cd paper && pdflatex main.tex || true

all: train eval latex

test:
	pytest tests/ -v

clean:
	rm -rf runs/ paper/figures/*.png paper/results.tex paper/*.aux paper/*.log paper/*.out
