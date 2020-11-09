.PHONY: install, test, docs

install:
	pip3 install .

test:
	python tests/runner.py

docs:
	sphinx-apidoc -f -o docs/src src
	cd docs && make html