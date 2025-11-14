import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SimpleCIFARNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    data_root = "./data/cifar10"

    train_full = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_full = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    # Small subsets so it runs fast
    train_subset = Subset(train_full, range(0, 2048))  # 2k samples
    test_subset = Subset(test_full, range(0, 512))     # 512 samples

    train_loader = DataLoader(
        train_subset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Model / loss / optim ---
    model = SimpleCIFARNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2
    print_every = 20

    print("Starting tiny CIFAR-10 debug run...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % print_every == 0:
                avg_loss = running_loss / n_batches
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )

            # hard cap: keep super fast
            if batch_idx >= 80:
                break

        # --- quick eval ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = 100.0 * correct / max(1, total)
        print(f"[Epoch {epoch}] Tiny test accuracy: {acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"Finished tiny CIFAR-10 debug run in {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
