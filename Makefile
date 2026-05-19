#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = music_genre_classifier
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
VENV ?= $(or $(UV_PROJECT_ENVIRONMENT),$(VIRTUAL_ENV),.venv)
export UV_PROJECT_ENVIRONMENT := $(VENV)
VENV_ACTIVATE = $(VENV)/bin/activate
UV_RUN = . "$(VENV_ACTIVATE)" && uv run --active

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


.PHONY: test
test:
	uv run pytest


## Prepare the XMIDI dataset
.PHONY: data
data:
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Virtual environment not found: $(VENV)"; \
		echo "Create it with: make create_environment VENV=/path/to/venv"; \
		exit 1; \
	fi
	$(UV_RUN) python scripts/prepare_data.py


## Train the Transformer neural classifier
.PHONY: train_transformer
train_transformer:
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Virtual environment not found: $(VENV)"; \
		echo "Create it with: make create_environment VENV=/path/to/venv"; \
		exit 1; \
	fi
	$(UV_RUN) python scripts/train_neural.py model=transformer


## Train the MuSeReNet neural classifier
.PHONY: train_muserenet
train_muserenet:
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Virtual environment not found: $(VENV)"; \
		echo "Create it with: make create_environment VENV=/path/to/venv"; \
		exit 1; \
	fi
	$(UV_RUN) python scripts/train_neural.py model=muserenet


## Fine-tune the Hugging Face MusicBERT genre classifier
.PHONY: train_musicbert
train_musicbert:
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Virtual environment not found: $(VENV)"; \
		echo "Create it with: make create_environment VENV=/path/to/venv"; \
		exit 1; \
	fi
	$(UV_RUN) python scripts/train_neural.py model=musicbert


## Evaluate frozen MusicBERT embeddings with a logistic-regression head
.PHONY: eval_musicbert_embeddings
eval_musicbert_embeddings:
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Virtual environment not found: $(VENV)"; \
		echo "Create it with: make create_environment VENV=/path/to/venv"; \
		exit 1; \
	fi
	$(UV_RUN) python scripts/evaluate_musicbert_embeddings.py model=musicbert



## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv "$(VENV)" --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: $(VENV)\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source $(VENV_ACTIVATE)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
