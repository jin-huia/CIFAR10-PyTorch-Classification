# CIFAR-10 Image Classification
PyTorch implementation for CIFAR-10 image classification task.

## Project Overview
This project builds a convolutional neural network for image classification on the CIFAR-10 dataset.
The code records training loss, training accuracy, test accuracy, precision and recall during training. After training completes, it automatically generates a 2×2 visualization curve plot to observe the model convergence.

## Environment
- Python >= 3.9
- PyTorch >= 2.0
- torchvision
- matplotlib
- tqdm
- scikit-learn
- pyyaml

Install dependencies:
```bash
pip install torch torchvision matplotlib tqdm scikit-learn
Project Structure
Classification_test/
├── train.py                 # Main training script
├── cifar10_dataset.py       # CIFAR10 dataset and dataloader
├── get_model.py             # Convolutional network definition
├── get_loss_fn.py           # Loss function definition
├── train_config.yaml        # Training hyperparameter config
├── tools/
│   └── utils.py             # Visualization & plotting functions
├── experiments/             # Output folder for saving training curve figures
└── README.md
Quick Start
Create and activate conda environment (optional)
conda create -n cifar10 python=3.13
conda activate cifar10
Install required packages
pip install torch torchvision matplotlib tqdm scikit-learn
Run training
python train.py
Features
Automatic GPU acceleration if CUDA is available, fallback to CPU automatically
Adam optimizer with StepLR learning rate scheduler. Learning rate multiplied by 0.5 every 5 epochs
Logged metrics: training loss, train accuracy, test accuracy, test precision, test recall
Automatically plot and save training curves to experiments/ folder
Dataset
CIFAR-10 dataset contains 50,000 training images and 10,000 test images with 10 object categories.
The dataset will be downloaded automatically on the first run.
Results
After training finishes, all_curves.png is generated under the experiments directory.
The figure includes four subplots:
Training Loss Curve
Train & Test Accuracy Curve
Test Precision Curve
Test Recall Curve
Model Analysis
The model shows healthy fitting performance. Training loss decreases steadily, while train and test accuracy rise synchronously with a small gap, which means no obvious underfitting or overfitting.
Because random augmentation is applied on training images, training samples are harder than test samples, resulting in slightly higher test accuracy than training accuracy. The model gradually converges at the end of training.
License
MIT