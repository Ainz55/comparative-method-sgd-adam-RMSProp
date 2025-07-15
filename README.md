# Overview

1. ### This project implements a Convolutional Neural Network (CNN) to compare the performance of different optimizers (SGD, Adam, RMSprop) on image classification tasks using MNIST and CIFAR10 datasets. Key features include:

   - Customizable CNN architecture with dropout and L2 regularization
   - Support for multiple datasets (MNIST, CIFAR10, FashionMNIST)
   - Optimizer comparison framework with training metrics tracking
   - Automatic learning rate scheduling
   - Visualization of training loss and test accuracy

### Requirements

- Python 3.7+
- Dependencies:
``torch torchvision numpy tqdm matplotlib``
___
### Key Configurable Parameters (in __main__ block)

- Optimizers: SGD, Adam, RMSprop
- Datasets: MNIST, CIFAR10
- Batch sizes: 32, 64
- Regularization:
  - Dropout: Enabled/disabled
  - L2 lambda: 0.001
- Training:
  - Epochs: 15
  - Learning rate: 0.001
  - Momentum: 0.9 (for SGD)
___
### Code Structure

1. Model Architecture (ImprovedCNN class):
   - Two convolutional layers with ReLU and max-pooling
   - Dynamic linear layer sizing
   - Optional dropout and L2 regularization
   - Custom regularization loss calculation
2. Dataset Loading (load_dataset function):
   - Automatic normalization based on dataset
   - Supports MNIST, CIFAR10, FashionMNIST
   - Returns DataLoaders with specified batch size

3. Training Pipeline (train_model function):
   - Configurable optimizer selection
   - StepLR learning rate scheduler (gamma=0.1 every 5 epochs)
   - Progress bars using tqdm
   - Tracks training loss and test accuracy per epoch

4. Visualization (plot_results function):
   - Generates side-by-side plots for:
     - Training loss vs epochs
     - Test accuracy vs epochs
   - Compares all optimizers in a single view
____
### Output

- Console output includes:
    - Device detection (CPU/GPU)
    - Per-epoch test accuracy
    - Progress bars for training batches

### Visualization:
<img width="1198" height="572" alt="image_2025-04-18_03-39-59" src="https://github.com/user-attachments/assets/274b31f8-1c68-49bd-ad76-9e740d5d8dc2" />
<img width="1196" height="572" alt="image_2025-04-18_03-40-23" src="https://github.com/user-attachments/assets/6c565cd9-1de2-40de-8459-467d422c4935" />



