.POSIX:
.PHONY: all test lint format clean

all: lint test format

test:
	python3 test_monkey.py

# Flake8 Ignored Errors:
#   E203 - Conflicts with black.
#   E221 - Disabled for manual vertically-aligned code.
#   E241 - Disabled for manual vertically-aligned code.
#   W503 - Conflicts with black.
lint:
	python3 -m mypy *.py
	python3 -m flake8 *.py --ignore=E203,E221,E241,W503

format:
	python3 -m black *.py --line-length 79

clean:
	rm -rf __pycache__/
	rm -rf .mypy_cache/
