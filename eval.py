import torch
from torch import nn
from tqdm import tqdm

from dataset.tiny_imagenet import get_dataloaders
from models.custom_net import CustomNet


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


def main():
    data_dir = "data/tiny-imagenet-200"
    checkpoint_path = "checkpoints/best_model.pth"
    batch_size = 64
    num_classes = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    _, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2
    )

    model = CustomNet(num_classes=num_classes).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device
    )

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation accuracy during training: {checkpoint['best_val_acc']:.2f}%")
    print(f"Evaluation validation loss: {val_loss:.4f}")
    print(f"Evaluation validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main()