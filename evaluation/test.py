import os
import sys
# 项目根目录，test.py在evaluation里面，往上跳一层
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
from datasets.cifar10_dataset import get_cifar10_loaders
from models.resnet18 import get_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar10_loaders("./data", 64)
    model = get_model().to(device)

    weight_path = os.path.join(BASE_DIR, "experiments", "model.pth")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            pred = torch.argmax(out, dim=1)
            correct += (pred == labels).sum().item()
            total += imgs.shape[0]

    print(f"test acc: {correct / total:.4f}")


if __name__ == "__main__":
    main()
