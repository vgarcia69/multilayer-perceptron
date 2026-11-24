VENV	   = venv
PYTHON	   = $(VENV)/bin/python
PIP        = $(VENV)/bin/pip
STAMP	   = $(VENV)/.installed
DATA      ?= data/data.csv
MODELS    ?= $(wildcard data/history_*.json)
TEST_FILE ?= data/validation.csv
CONFIG    ?= data/config.json
GENERATED  = data/training.csv data/validation.csv data/trained_model.json

.PHONY: split train predict compare clean fclean echo

echo:
	@echo "Available targets:"
	@echo "  split   - Split the dataset (make split DATA=path/to/file.csv)"
	@echo "  train   - Train the model using the training set"
	@echo "  predict - Make predictions using the trained model"
	@echo "  compare - Compare training curves (make compare MODELS=\"a.json b.json\")"
	@echo "  clean   - Remove generated files"
	@echo "  fclean  - Remove generated files and virtual environment"

split: $(STAMP)
	$(PYTHON) srcs/split_data.py $(DATA)

train: $(STAMP)
	$(PYTHON) srcs/train.py $(CONFIG)

predict: $(STAMP)
	$(PYTHON) srcs/predict.py $(TEST_FILE)

compare: $(STAMP)
	$(PYTHON) srcs/compare.py $(MODELS)

$(STAMP):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install numpy pandas matplotlib colorama
	touch $(STAMP)

clean:
	rm -f $(GENERATED)

fclean: clean
	rm -rf $(VENV)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
