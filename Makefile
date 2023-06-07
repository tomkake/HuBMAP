.PHONY: black-check
black-check:
	rye run black --check src 

.PHONY: black
black:
	rye run black src 

.PHONY: flake8
flake8:
	rye run flake8 src 

.PHONY: isort-check
isort-check:
	rye run isort --check-only src 

.PHONY: isort
isort:
	rye run isort src 

.PHONY: mdformat
mdformat:
	rye run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	rye run mdformat --check *.md

.PHONY: mypy
mypy:
	rye run mypy src

.PHONY: test
test:
	rye run pytest  --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort
	$(MAKE) mdformat

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) mdformat-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) lint
	$(MAKE) test