.PHONY: train evaluate predict test

EPOCHS ?= 5
LEARNING_RATE ?= 0.001
DEVICE ?= auto
MODEL_NAME ?=
IMAGE_PATH ?=

train:
	@uv run -m src.lab1.modeling.train \
		--epochs $(EPOCHS) \
		--lr $(LEARNING_RATE) \
		$(if $(filter-out auto,$(DEVICE)),--device $(DEVICE),)

evaluate:
	@uv run -m src.lab1.modeling.evaluate \
		--model-name $(MODEL_NAME) \
		$(if $(filter-out auto,$(DEVICE)),--device $(DEVICE),)

predict:
	@uv run -m src.lab1.modeling.predict \
		--model-name $(MODEL_NAME) \
		--image-path $(IMAGE_PATH) \
		$(if $(filter-out auto,$(DEVICE)),--device $(DEVICE),)

# Run tests
test:
	@echo "Running tests..."
	@uv run pytest tests/ -v
