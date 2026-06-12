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
