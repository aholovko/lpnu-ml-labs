EPOCHS ?= 5
LEARNING_RATE ?= 0.001
DEVICE ?= auto
MODEL_NAME ?=
IMAGE_PATH ?=
LAB ?= 2

.PHONY: train
train:
	@uv run -m src.lab$(LAB).train \
		--epochs $(EPOCHS) \
		--lr $(LEARNING_RATE) \
		$(if $(filter-out auto,$(DEVICE)),--device $(DEVICE),)

.PHONY: predict
predict:
	@uv run -m src.lab1.predict \
		--model-name $(MODEL_NAME) \
		--image-path $(IMAGE_PATH) \
		$(if $(filter-out auto,$(DEVICE)),--device $(DEVICE),)

.PHONY: test
test:
	@echo "Running tests..."
	@uv run pytest tests/ -v

.PHONY: checks
checks: uv-lock lint format typecheck

.PHONY: uv-lock
uv-lock:
	@echo "Locking dependencies..."
	@uv lock

.PHONY: lint
lint:
	@echo "Linting code..."
	@uv run ruff check --fix .

.PHONY: format
format:
	@echo "Formatting code..."
	@uv run ruff format .

.PHONY: typecheck
typecheck:
	@echo "Type checking code..."
	@uv run pyright
