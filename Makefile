.PHONY: install test wheel doctor

install:
	python -m pip install -e '.[runtime]'

test:
	python -m pytest tests/test_runtime_cli.py -q

wheel:
	python -m build --wheel

doctor:
	wal-runtime doctor
