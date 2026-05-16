import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_datasets(data_dir="data/tiny-imagenet-200"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val_images_by_class")

    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform
    )

    return train_dataset, val_dataset


def get_dataloaders(data_dir="data/tiny-imagenet-200", batch_size=64, num_workers=2):
    train_dataset, val_dataset = get_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def load_class_names(data_dir="data/tiny-imagenet-200"):
    words_path = os.path.join(data_dir, "words.txt")

    wnid_to_name = {}

    if not os.path.exists(words_path):
        return wnid_to_name

    with open(words_path, "r") as f:
        for line in f:
            wnid, name = line.strip().split("\t")
            wnid_to_name[wnid] = name.split(",")[0]

    return wnid_to_name


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(
        data_dir="data/tiny-imagenet-200",
        batch_size=64,
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Number of classes:", len(train_loader.dataset.classes))