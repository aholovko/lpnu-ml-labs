.PHONY: train evaluate clean

BATCH_SIZE ?= 64
EPOCHS ?= 5
LR ?= 0.001
DROPOUT ?= 0.5
DATA_DIR ?= ./data
MODEL_PATH ?= models/mnist-cnn.pt
PLOT_PATH ?= training_history.png
SEED ?= 1
DEVICE ?= 

# Train the model
train:
	@echo "Training CNN model on MNIST dataset..."
	@uv run -m lab1.train \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--dropout $(DROPOUT) \
		--data-dir $(DATA_DIR) \
		--model-path $(MODEL_PATH) \
		--plot-path $(PLOT_PATH) \
		--seed $(SEED)

# Evaluate the trained model
evaluate:
	@echo "Evaluating CNN model on MNIST test dataset..."
	@uv run -m lab1.evaluate \
		--batch-size $(BATCH_SIZE) \
		--data-dir $(DATA_DIR) \
		--model-path $(MODEL_PATH) \
		--output-file evaluation_results.txt \
		--seed $(SEED) \
		$(if $(DEVICE),--device $(DEVICE),)

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	@rm -f $(PLOT_PATH) evaluation_results.txt
	@rm -f $(MODEL_PATH)
