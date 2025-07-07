import torch
import torch.nn as nn

class Generator(nn.Module):
  """
  Generator model for the GAN.
  Takes a noise vector and generates an image.
  """
  def __init__(self, noise_dim, image_dim):
    """
    Initializes the Generator.

    Args:
      noise_dim (int): Dimension of the input noise vector.
      image_dim (int): Dimension of the output image (e.g., 28*28*1 for Fashion-MNIST).
    """
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(noise_dim, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), # Dropout added for regularization
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.25), 
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), 
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.25), 
        nn.Linear(256, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2), 
        nn.Linear(256, image_dim),
        nn.Tanh() # Tanh scales output to [-1, 1]
    )
    # self.image_dim = image_dim 

  def forward(self, z):
    """
    Forward pass for the Generator.

    Args:
      z (torch.Tensor): Input noise vector.

    Returns:
      torch.Tensor: Generated image tensor.
    """
    # Reshape the output from flat vector to image dimensions (batch_size, channels, height, width)
    # return self.model(z).view(z.size(0), 1, 28, 28)
    return self.model(z)

