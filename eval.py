import argparse
import torch
from torch import nn
from tqdm import tqdm

from dataset.tiny_imagenet import get_dataloaders
from models.model_factory import get_model


def evaluate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Tiny ImageNet model checkpoint."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file to evaluate."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tiny-imagenet-200",
        help="Path to Tiny ImageNet dataset."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint["config"]
    model_name = checkpoint["model_name"]
    num_classes = config.get("num_classes", 200)

    print(f"Model name: {model_name}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy during training: {checkpoint['best_val_acc']:.2f}%")

    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )

    model = get_model(
        model_name=model_name,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device
    )

    print(f"Evaluation validation loss: {val_loss:.4f}")
    print(f"Evaluation validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main()