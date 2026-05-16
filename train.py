import argparse
import os
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import wandb

from dataset.tiny_imagenet import get_dataloaders
from models.model_factory import get_model
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


def build_optimizer(model, optimizer_name, learning_rate, momentum, weight_decay):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        "Available optimizers: SGD, Adam"
    )


def make_run_name(model, optimizer, lr, batch_size, epochs, weight_decay):
    lr_str = str(lr).replace(".", "p")
    wd_str = str(weight_decay).replace(".", "p")

    return (
        f"{model.lower()}_"
        f"{optimizer.lower()}_"
        f"lr{lr_str}_"
        f"bs{batch_size}_"
        f"wd{wd_str}"
    )


def save_experiment_results(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    results_df = pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"]
    })

    results_df.to_csv(save_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train different CNN models on Tiny ImageNet."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="customnet",
        choices=["customnet", "alexnet", "resnet18"],
        help="Model architecture to train."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam"],
        help="Optimizer to use."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay regularization."
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum value for SGD."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional custom Wandb run name."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tiny-imagenet-200",
        help="Path to Tiny ImageNet dataset."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the matching checkpoint if it exist"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = args.run_name

    if run_name is None:
        run_name = make_run_name(
            model=args.model,
            optimizer=args.optimizer,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=args.weight_decay
        )

    config = {
        "data_dir": args.data_dir,
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "num_classes": 200,
        "optimizer": args.optimizer,
        "loss_function": "CrossEntropyLoss",
        "model": args.model,
        "run_name": run_name,
        "resume": args.resume
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    run_results_dir = os.path.join(config["results_dir"], run_name)
    os.makedirs(run_results_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        config["checkpoint_dir"],
        f"{run_name}_best.pth"
    )

    wandb.init(
        project="faimdl-lab3-tiny-imagenet",
        name=run_name,
        id=run_name,
        resume="allow" if args.resume else None,
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

    model = get_model(
        model_name=config["model"],
        num_classes=config["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model=model,
        optimizer_name=config["optimizer"],
        learning_rate=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )

    wandb.watch(model, criterion, log="all", log_freq=100)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    start_epoch = 1

    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy so far: {best_val_acc:.2f}%")

    elif args.resume and not os.path.exists(checkpoint_path):
        print("Resume was requested, but no matching checkpoint was found.")
        print("Starting training from scratch.")

    if start_epoch > config["num_epochs"]:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is equal to or greater than target epochs={config['num_epochs']}."
        )
        print("Nothing to train.")
        wandb.finish()
        return

    for epoch in range(start_epoch, config["num_epochs"] + 1):
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

            torch.save({
                "epoch": epoch,
                "model_name": config["model"],
                "model_state_dict": model.state_dict(),
                "optimizer_name": config["optimizer"],
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
                "config": config
            }, checkpoint_path)

            wandb.save(checkpoint_path)

            print(f"Best model saved to {checkpoint_path}")

    plot_training_curves(history, save_dir=run_results_dir)

    save_experiment_results(
        history,
        save_path=os.path.join(run_results_dir, "experiment_results.csv")
    )

    wandb.log({
        "best_val_accuracy": best_val_acc
    })

    wandb.finish()

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Best checkpoint saved at: {checkpoint_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()