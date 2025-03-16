# LPNU ML Labs

## Requirements

- Python 3.12
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)

## Project Structure

```
├── data/               <- Directory for storing data
├── models/             <- Trained models
├── src/
│   └── lab1/           <- Source code for lab1
│       ├── config.py   <- Configuration variables
│       ├── dataset.py  <- Dataset module
│       ├── features.py <- Feature engineering 
│       ├── utils.py    <- Utility functions
│       ├── modeling/   <- Model training and inference
│       │   ├── model.py    <- Model definition
│       │   ├── train.py    <- Model training
│       │   ├── evaluate.py <- Model evaluation
│       │   └── predict.py  <- Model inference
└── README.md           <- This file
```

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/aholovko/lpnu-ml-labs.git
cd lpnu-ml-labs
```

### 2. Set up the environment

```sh
uv sync
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

### 3. Run model training

Start the model training by running:

```sh
make train
```
