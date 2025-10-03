paper:
	cd paper && pdflatex main.tex || true
train:
	python -m hchealth.train --config configs/clinical_demo.yaml
eval:
	python -m hchealth.evaluate --config configs/clinical_demo.yaml
latex:
	python -m hchealth.to_latex --config configs/clinical_demo.yaml
all:
	make train && make eval && make latex && make paper
