-include .env

ifeq ($(OS),Windows_NT)
    PYTHON := venv/Scripts/python
else
    PYTHON := venv/bin/python
endif

.PHONY:install download_datasets data train predict_validation local_deploy publish_model

install:
	python -c "import os, shutil; shutil.copy('.env.example', '.env') if not os.path.exists('.env') else print('.env already exists')"
	python -m venv venv
	$(PYTHON) -m pip install -r models_requirements.txt

download_datasets:
	$(PYTHON) src/predicting_transportation_costs/download_data.py

data:
	$(PYTHON) src/predicting_transportation_costs/make_data.py

train:
	$(PYTHON) src/predicting_transportation_costs/train.py

predict_validation:
	$(PYTHON) src/predicting_transportation_costs/predict.py

test: 
	$(PYTHON) -m pytest tests/


local_deploy:
	docker build --build-arg MODEL_VERSION=$(model_version) -t predicting-transportation-costs .
	docker run --rm --env-file .env -p 8000:8000 predicting-transportation-costs

publish_model:
	$(PYTHON) src/predicting_transportation_costs/publish_model.py