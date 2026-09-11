include mlops/mk/physionet.mk
include mlops/mk/data.mk

PHONY += clean
clean:
	rm -rf ${VENV}
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	uv cache prune

PHONY += install
install:
	uv run dvc repro

PHONY += update
update:
	uv lock --upgrade
	uv sync

PHONY += test
test:
	uv run --locked --with-requirements test/requirements.txt pytest

PHONY += test-smoke
test-smoke:
	uv run --locked --with-requirements test/requirements.txt pytest -m smoke

PHONY += test-cov
test-cov:
	uv run --locked --with-requirements test/requirements.txt pytest --cov=realphe --cov-report=term-missing

.PHONY: ${PHONY}
