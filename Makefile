.POSIX:
.PHONY: all test lint clean format

all: test

test:
	python3 test_monkey.py

lint:
	python3 -m mypy *.py

clean:
	rm -rf __pycache__/

format:
	black *.py --line-length 80
