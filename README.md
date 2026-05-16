# FAIMDL Lab 3 - Tiny ImageNet Custom Model Project

This repository contains the Lab 3 project for the Fundamentals of Artificial Intelligence, Machine and Deep Learning course.

The goal of this lab is to transform the previous Lab 2 notebook-based solution into a structured PyTorch project using GitHub, pip, and Weights & Biases.

## Project Structure

```text
faimdl-lab3/
│
├── checkpoints/              # Saved model checkpoints
├── data/                     # Dataset folder, not pushed to GitHub
├── dataset/
│   └── tiny_imagenet.py      # Dataset, transforms, dataloaders
├── models/
│   └── custom_net.py         # Custom CNN model
├── utils/
│   ├── download_dataset.py   # Dataset download and validation organization
│   └── visualization.py      # Training curves and helper visualization functions
├── results/                  # Training results, curves and CSV files
├── train.py                  # Training script
├── eval.py                   # Evaluation script
├── requirements.txt          # Required Python packages
└── README.md