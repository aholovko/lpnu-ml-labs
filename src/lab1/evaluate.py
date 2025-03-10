import torch
import argparse
import os
from src.lab1.dataloader import MNISTDataModule


def evaluate_model(model, test_dataloader, device=None):
    """
    Evaluate the model on the test dataset.

    Args:
        model: The trained model to evaluate
        test_dataloader: DataLoader for the test dataset
        device: Device to run the evaluation on (CPU or GPU)

    Returns:
        test_accuracy: The accuracy on the test dataset
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in test_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            pred = model(x_batch)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            total_correct += is_correct.sum().item()
            total_samples += y_batch.size(0)

    test_accuracy = total_correct / total_samples
    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN model on MNIST dataset"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="directory to store the dataset (default: ./data)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mnist-cnn.pt",
        help="path to the trained model (default: models/mnist-cnn.pt)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="file to save evaluation results (optional)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use for evaluation (default: auto-detect)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model file {args.model_path} not found. Please train the model first."
        )

    model = torch.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Initialize the data module
    data_module = MNISTDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.prepare_data()
    data_module.setup()

    # Get test dataloader
    test_dl = data_module.test_dataloader()

    # Evaluate model
    print("Evaluating model on test dataset...")
    test_accuracy = evaluate_model(model, test_dl, device=device)

    # Save results to file if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(f"Test accuracy: {test_accuracy:.4f}\n")
        print(f"Evaluation results saved to {args.output_file}")

    print("Evaluation completed successfully!")
