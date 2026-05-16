import torch
from torch import nn
from torchvision import models


def get_alexnet(num_classes=200):
    """
    Returns an AlexNet model adapted for Tiny ImageNet.
    Tiny ImageNet has 200 classes, so the final classifier layer is changed.
    """
    model = models.alexnet(weights=None)

    model.classifier[6] = nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )

    return model


if __name__ == "__main__":
    model = get_alexnet(num_classes=200)

    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)

    print(model)
    print("Output shape:", output.shape)