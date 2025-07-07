# Fashion-MNIST GAN

This repository contains a Generative Adversarial Network (GAN) implemented in PyTorch for generating images similar to the Fashion-MNIST dataset. It provides a flexible command-line interface for training and generating fashion-like images.

---

## Project Structure

```
fashion-mnist-gan/
├── data/
│   └── fashion-mnist_train.csv  
├── models/
│   ├── generator.py
│   └── discriminator.py
├── utils/
│   ├── data_loader.py
│   └── visualize.py
├── checkpoints/                 (This directory will be created during training)
│   └── generator_epoch_XXX.pth
│   └── discriminator_epoch_XXX.pth
├── results/                     (This directory will be created during training)
│   └── epoch_XXX.png
├── main.py     
├── train.py
├── generate.py
├── README.md
└── requirements.txt
```
---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Indiandude123/GAN_on_FashionMNIST.git 
cd fashion-mnist-gan
```

### 2. Prepare the Dataset

Download `fashion-mnist_train.csv` from a source like [Kaggle - Fashion MNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and place it inside the `data/` directory.

### 3. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All functionality is accessed through the `main.py` interface.

---

### Training the GAN

To train the GAN:

```bash
python main.py train
```

#### Available Training Arguments:

| Argument         | Type  | Default    | Description                                                                 |
| ---------------- | ----- | ---------- | --------------------------------------------------------------------------- |
| `--epochs`       | int   | 200        | Number of training epochs                                                   |
| `--batch_size`   | int   | 128        | Batch size for training                                                     |
| `--mode`         | str   | "one\_one" | Training mode. Choices: `one_one`, `five_gen_one_disc`, `five_disc_one_gen` |
| `--lr`           | float | 0.000086   | Learning rate for Adam optimizer                                            |
| `--weight_decay` | float | 0.00001    | Weight decay (L2 regularization) for Adam optimizer                         |

#### Examples:

```bash
# Train with default settings
python main.py train

# Train for 150 epochs with 5 generator steps per discriminator step
python main.py train --epochs 150 --mode five_gen_one_disc

# Train using a custom learning rate
python main.py train --lr 0.0001
```

> Model checkpoints are saved in the `checkpoints/` directory.
> Generated images during training are saved in the `results/` directory.

---

### Generating Samples from a Trained Model

To generate samples from a trained generator model:

```bash
python main.py generate
```

#### Available Generation Arguments:

| Argument        | Type | Default                           | Description                               |
| --------------- | ---- | --------------------------------- | ----------------------------------------- |
| `--model_path`  | str  | `checkpoints/generator_final.pth` | Path to the trained generator `.pth` file |
| `--num_samples` | int  | 16                                | Number of samples to generate             |

#### Examples:

```bash
# Generate 25 samples from the final model
python main.py generate --num_samples 25

# Generate using a specific checkpoint
python main.py generate --model_path checkpoints/generator_epoch_100.pth
```

> Generated sample images will be saved inside the `results/` directory.

---

## Model Architecture

### Generator

* Type: Multi-Layer Perceptron (MLP)
* Input: Random noise vector
* Output: 28×28 grayscale image
* Activations: LeakyReLU (hidden layers), Tanh (output layer)

### Discriminator

* Type: Multi-Layer Perceptron (MLP)
* Input: 28×28 grayscale image
* Output: A single probability (real or fake)
* Activations: LeakyReLU (hidden layers), Sigmoid (output layer)

---

## Dependencies

Key libraries used in this project:

* `torch`
* `torchvision`
* `matplotlib`
* `pandas`
* `numpy`
* `scikit-learn`
* `Pillow`

To install all dependencies:

```bash
pip install -r requirements.txt
```

---
