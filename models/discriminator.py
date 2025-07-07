import torch
import torch.nn as nn

class Discriminator(nn.Module):
  """
  Discriminator model for the GAN.
  Takes an image and classifies it as real or fake.
  """
  def __init__(self, image_dim):
    """
    Initializes the Discriminator.

    Args:
      image_dim (int): Dimension of the input image (e.g., 28*28*1 for Fashion-MNIST).
    """
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(image_dim, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 1),
        nn.Sigmoid() # Sigmoid outputs a probability between 0 and 1
    )

  def forward(self, image):
    """
    Forward pass for the Discriminator.

    Args:
      image (torch.Tensor): Input image tensor.

    Returns:
      torch.Tensor: Probability that the image is real.
    """
    # Flatten the image for input to the linear layers
    # image = image.view(image.size(0), -1)
    return self.model(image)

