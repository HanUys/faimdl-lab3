import os
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import wandb

from dataset.tiny_imagenet import get_dataloaders
from models.custom_net import CustomNet
from utils.visualization import plot_training_curves


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total

        progress_bar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "acc": f"{current_acc:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total

    return val_loss, val_acc


def save_experiment_results(history, save_path="results/experiment_results.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results_df = pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"]
    })

    results_df.to_csv(save_path, index=False)


def main():
    config = {
        "data_dir": "data/tiny-imagenet-200",
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
        "batch_size": 64,
        "num_epochs": 5,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "num_classes": 200,
        "optimizer": "SGD",
        "loss_function": "CrossEntropyLoss",
        "model": "CustomNet"
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    wandb.init(
        project="faimdl-lab3-tiny-imagenet",
        name="customnet_sgd_lr0001_epoch5",
        config=config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=2
    )

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")

    model = CustomNet(num_classes=config["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"]
    )

    wandb.watch(model, criterion, log="all", log_freq=100)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(1, config["num_epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint_path = os.path.join(
                config["checkpoint_dir"],
                "best_model.pth"
            )

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
                "config": config
            }, checkpoint_path)

            wandb.save(checkpoint_path)

            print(f"Best model saved to {checkpoint_path}")

    plot_training_curves(history, save_dir=config["results_dir"])

    save_experiment_results(
        history,
        save_path=os.path.join(config["results_dir"], "experiment_results.csv")
    )

    wandb.log({
        "best_val_accuracy": best_val_acc
    })

    wandb.finish()

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print("Training completed.")


if __name__ == "__main__":
    main()