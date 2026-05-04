.PHONY: install setup run-ui test docker-build docker-run clean

install:
	./setup.sh

setup: install

run-ui:
	source .venv/bin/activate && python3 app.py

test:
	source .venv/bin/activate && PYTHONPATH=. pytest tests/

benchmark:
	source .venv/bin/activate && python3 benchmark.py

check-gpu:
	source .venv/bin/activate && python3 check_gpu.py

docker-build:
	docker-compose build

docker-run:
	docker-compose up

clean:
	rm -rf separated/ gfpgan/ __pycache__/ .pytest_cache/
	rm -f temp_* transformed_output.mp4 final_*.mp4
