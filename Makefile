.PHONY: docs
.PHONY: tests

install:
	pip3 install .

tests:
	pylint -j 0 --disable=R,C,W src/datascience_starter
	python tests/runner.py

docs:
	sphinx-apidoc -f -o docs/src src
	cd docs && make html

lint:
	pylint src/datascience_starter