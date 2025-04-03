# LPNU ML Labs

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checker: pyright](https://img.shields.io/badge/type%20checker-pyright-3775A9.svg)](https://github.com/microsoft/pyright)

## Requirements

- Python 3.12
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

## Project Structure

```
├── data/                          <- Datasets
├── models/                        <- Trained models
├── notebooks/                     <- Jupyter notebooks
├── reports/                       <- Generated reports
│   └── figures/                   <- Generated figures
├── src/
│   ├── __init__.py
│   ├── datamodule.py              <- Base data module
│   ├── model.py                   <- Base model class
│   ├── paths.py                   <- Path definitions
│   ├── trainer.py                 <- Trainer class
│   ├── utils.py                   <- Utility functions
│   ├── lab1/
│   │   ├── __init__.py
│   │   ├── config.py              <- Options for model training
│   │   ├── dataset_mnist.py       <- Data module for MNIST dataset
│   │   ├── model.py               <- Model definition
│   │   ├── predict.py             <- Model inference
│   │   └── train.py               <- Model training
│   └── lab2/
│       ├── __init__.py
│       ├── config.py              <- Options for model training
│       ├── dataset_audio_mnist.py <- Data module for Audio MNIST dataset
│       ├── model.py               <- Model definition
│       └── train.py               <- Model training
├── tests/                         <- Unit tests
├── LICENSE
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aholovko/lpnu-ml-labs.git
cd lpnu-ml-labs
```

### 2. Set up the environment

```bash
uv sync
```

Activate the virtual environment:
```bash
# On Unix/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## Usage

### Train the Model

Choose the lab and start model training with default parameters:

#### Lab 1 (MNIST Digit Recognition):

```bash
make train LAB=1
```
or
```bash
python -m src.lab1.train
```

#### Lab 2 (Audio MNIST):

```bash
make train LAB=2
```
or
```bash
python -m src.lab2.train
```

Customize training parameters:

```bash
make train LAB=1 EPOCHS=10 LEARNING_RATE=0.0005 DEVICE=cuda
```
or
```bash
python -m src.lab1.train --epochs 10 --lr 0.0005 --device cuda
```

### Make Predictions

Generate predictions using a trained model:

```bash
make predict LAB=1 MODEL_NAME=model_name IMAGE_PATH=path/to/image.jpg
```
or
```bash
python -m src.lab1.predict --model-name model_name --image-path path/to/image.jpg
```

> **Note**: `MODEL_NAME` should be the filename (without extension) of a model file located in the `./models/` directory.

## Development

### Code Quality Checks

Run all code quality checks with:

```bash
make checks
```

This command runs the following checks in sequence:

1. **Dependencies**: `make uv-lock` - Locks dependencies
2. **Linting**: `make lint` - Lints the code using Ruff with auto-fixes 
3. **Formatting**: `make format` - Formats code using Ruff formatter
4. **Type checking**: `make typecheck` - Performs static type checking with Pyright

You can also run each check individually as needed.

### Run Tests

Run the test suite with:

```bash
make test
```
or
```bash
python -m pytest tests/ -v
```

### Run Jupyter Notebook

```bash
uv run --with jupyter jupyter lab
```

## Contributing

Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

- **CUDA not found**: Make sure you have the appropriate CUDA drivers installed if using `DEVICE=cuda`
- **Model not found**: Ensure the model file exists in the `./models/` directory
- **Import errors**: Verify your virtual environment is activated
