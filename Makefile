.PHONY: install dev compile ask eval report serve mcp test lint fmt docker

install:
	python -m pip install -e ".[docs]"

dev:
	python -m pip install -e ".[dev,docs,langchain]"

compile:
	fde compile --reset

ask:
	fde ask "$(q)" --trace

eval:
	fde eval

report:
	fde report

serve:
	fde serve --port 8000

mcp:
	fde mcp

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

docker:
	docker compose up --build
