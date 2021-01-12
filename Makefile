.POSIX:
.PHONY: all test clean format

all: test

test:
	python3 test_monkey.py

clean:
	rm -rf __pycache__/

format:
	black *.py --line-length 80
