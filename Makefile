.POSIX:
.PHONY: all test lint format clean

all: test lint format

test:
	python3 test_monkey.py

lint:
	python3 -m mypy *.py

format:
	black *.py --line-length 80

clean:
	rm -rf __pycache__/
	rm -rf .mypy_cache/
