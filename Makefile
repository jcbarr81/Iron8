PY=python

.PHONY: setup serve test train calibrate export-onnx verify-artifacts fmt lint

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

serve:
	uvicorn run_server:app --reload --port 8000

test:
	pytest -q

train:
	$(PY) scripts/train_pa.py

calibrate:
	$(PY) scripts/calibrate_pa.py

export-onnx:
	$(PY) scripts/export_onnx.py

verify-artifacts:
	$(PY) scripts/check_artifacts.py

fmt:
	$(PY) -m pip install ruff black
	ruff check --fix . || true
	black .

lint:
	ruff check . || true
