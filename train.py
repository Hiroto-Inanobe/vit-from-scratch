import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from models.cnn import SimpleCNN
from models.vit import VisionTransformer


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model(model, train_loader, test_loader, name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []

    for epoch in range(30):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        avg_loss = total_loss / len(train_loader)

        losses.append(avg_loss)
        accs.append(acc)

        print(f"{name} | epoch {epoch+1} | loss {avg_loss:.4f} | acc {acc:.4f}")

    # save plots
    plt.figure()
    plt.plot(losses)
    plt.title(f"{name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"results/{name}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(accs)
    plt.title(f"{name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"results/{name}_acc.png")
    plt.close()

    return losses, accs


def main():
    transform = transforms.ToTensor()

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    print("=== CNN ===")
    train_model(SimpleCNN(), train_loader, test_loader, "CNN")

    print("=== ViT ===")
    train_model(VisionTransformer(), train_loader, test_loader, "ViT")


if __name__ == "__main__":
    main()
