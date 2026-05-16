import torch
from torch import nn
from torchvision import models


def get_resnet18(num_classes=200):
    """
    Returns a ResNet18 model adapted for Tiny ImageNet.
    Tiny ImageNet has 200 classes, so the final fully connected layer is changed.
    """
    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model


if __name__ == "__main__":
    model = get_resnet18(num_classes=200)

    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)

    print(model)
    print("Output shape:", output.shape)