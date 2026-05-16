from models.custom_net import CustomNet
from models.alexnet import get_alexnet
from models.resnet import get_resnet18


def get_model(model_name, num_classes=200):
    model_name = model_name.lower()

    if model_name == "customnet":
        return CustomNet(num_classes=num_classes)

    if model_name == "alexnet":
        return get_alexnet(num_classes=num_classes)

    if model_name == "resnet18":
        return get_resnet18(num_classes=num_classes)

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Available models: customnet, alexnet, resnet18"
    )

if __name__ == "__main__":
    
    import torch

    for model_name in ["customnet", "alexnet", "resnet18"]:
        model = get_model(model_name, num_classes=200)
        dummy_input = torch.randn(1, 3, 64, 64)
        output = model(dummy_input)

        print(f"{model_name} output shape:", output.shape)