.PHONY: test build clean lan_server lan_client

install:
	uv sync --extra dev

lan_server:
	uv run python examples/lan_server.py

lan_client:
	uv run python examples/lan_client.py

###################
# Developer tools #
###################

test_performance:
	uv run --extra dev python -m tests.test_performance

test:
	uv run --extra dev pytest

pc:
	uv run --extra dev ruff check .

build:
	uv build

clean:
	rm -rf build dist *.egg-info
