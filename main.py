import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ImprovedCNN(nn.Module):
    def __init__(self, input_channels=1, use_dropout=False, l2_lambda=0.0):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25) if use_dropout else nn.Identity()

        # Автоматический расчет размера
        self._to_linear = None
        self.fc1 = nn.Linear(1, 1)  # Временная инициализация
        self.fc2 = nn.Linear(128, 10)
        self.l2_lambda = l2_lambda

    def _calculate_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.pool(torch.relu(self.conv1(input)))
            output = self.pool(torch.relu(self.conv2(output)))
            return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self._to_linear)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def l2_regularization(self):
        if self.l2_lambda == 0:
            return 0
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_lambda * l2_loss


def load_dataset(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name in ["MNIST",
                                                                 "FashionMNIST"] else
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True,
                                              transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True,
                                             transform=transform)
        input_shape = (1, 28, 28)
    elif dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True,
                                                transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True,
                                               transform=transform)
        input_shape = (3, 32, 32)
    elif dataset_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(root='./data',
                                                     train=True,
                                                     download=True,
                                                     transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data',
                                                    train=False,
                                                    download=True,
                                                    transform=transform)
        input_shape = (1, 28, 28)
    else:
        raise ValueError("Unknown dataset")

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    return trainloader, testloader, input_shape


def train_model(optimizer_name, dataset_name, batch_size=64, lr=0.001,
                epochs=10, use_dropout=False, l2_lambda=0.0, momentum=0.9):
    trainloader, testloader, input_shape = load_dataset(dataset_name,
                                                        batch_size)

    model = ImprovedCNN(
        input_channels=input_shape[0],
        use_dropout=use_dropout,
        l2_lambda=l2_lambda
    )

    model._to_linear = model._calculate_conv_output(input_shape)
    model.fc1 = nn.Linear(model._to_linear, 128)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unknown optimizer")

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(trainloader,
                                   desc=f'{optimizer_name} | Epoch {epoch + 1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + model.l2_regularization()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        train_losses.append(running_loss / len(trainloader))

        # Тестирование
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Test Accuracy: {accuracy:.2f}%')

    return train_losses, test_accuracies


def plot_results(results, dataset, batch_size):
    plt.figure(figsize=(12, 5))
    key = f"{dataset}_bs{batch_size}"

    plt.subplot(1, 2, 1)
    for opt_name in results[key]:
        plt.plot(results[key][opt_name]['train_loss'], label=opt_name)
    plt.title(f'{dataset} (BS={batch_size}): Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for opt_name in results[key]:
        plt.plot(results[key][opt_name]['test_acc'], label=opt_name)
    plt.title(f'{dataset} (BS={batch_size}): Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    optimizers = ['SGD', 'Adam', 'RMSprop']
    datasets = ['MNIST', 'CIFAR10']
    batch_sizes = [32, 64]
    use_dropout = True
    l2_lambda = 0.001
    momentum = 0.9
    epochs = 15

    results = {}

    for dataset in datasets:
        for batch_size in batch_sizes:
            print(f"\n=== Dataset: {dataset}, Batch Size: {batch_size} ===")
            dataset_results = {}
            for opt_name in optimizers:
                print(f"\n--- Optimizer: {opt_name} ---")
                train_loss, test_acc = train_model(
                    opt_name, dataset, batch_size, lr=0.001, epochs=epochs,
                    use_dropout=use_dropout, l2_lambda=l2_lambda,
                    momentum=momentum
                )
                dataset_results[opt_name] = {'train_loss': train_loss,
                                             'test_acc': test_acc}
            results[f"{dataset}_bs{batch_size}"] = dataset_results

    # Визуализация результатов для CIFAR10 с batch_size=64
    if 'CIFAR10_bs64' in results:
        plot_results(results, "CIFAR10", 64)

    # Визуализация результатов для MNIST с batch_size=64
    if 'MNIST_bs64' in results:
        plot_results(results, "MNIST", 64)
