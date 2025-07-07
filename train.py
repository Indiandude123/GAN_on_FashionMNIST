import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.utils as vutils # For saving generated images

# Import models and utilities
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_fashion_mnist_dataloader
from utils.visualize import save_generated_images # Removed view_samples as it's for generate.py

# --- Configuration (defaults, can be overridden by main.py) ---
DATA_PATH = "data/fashion-mnist_train.csv"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_training(num_epochs, batch_size, mode, learning_rate, weight_decay):
  """
  Trains the GAN (Generator and Discriminator).

  Args:
    num_epochs (int): Number of epochs to train.
    batch_size (int): Batch size for the DataLoader.
    mode (str): Training mode ("one_one", "five_gen_one_disc", "five_disc_one_gen").
    learning_rate (float): Learning rate for optimizers.
    weight_decay (float): Weight decay for optimizers.
  """
  # --- Model Parameters ---
  NOISE_DIM = 100
  IMAGE_DIM = 28 * 28 * 1 # For Fashion-MNIST (1 channel, 28x28 pixels)

  # --- Model Instantiation ---
  generator = Generator(NOISE_DIM, IMAGE_DIM).to(device)
  discriminator = Discriminator(IMAGE_DIM).to(device)

  # --- Optimizers ---
  gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
  disc_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)

  # --- Loss Function ---
  criterion = nn.BCELoss() # Binary Cross-Entropy Loss for GANs

  # --- Create directories ---
  os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
  os.makedirs(RESULTS_DIR, exist_ok=True)

  # Get DataLoader
  train_loader = get_fashion_mnist_dataloader(DATA_PATH, batch_size)

  # Fixed noise for visualizing generator's progress over epochs
  fixed_noise = torch.randn(64, NOISE_DIM).to(device) # 64 samples for visualization

  history_disc_losses = []
  history_gen_losses = []

  print(f'Starting GAN training for {num_epochs} epochs in mode: {mode}')

  for epoch in range(num_epochs):
    epoch_disc_loss = 0.0
    epoch_gen_loss = 0.0
    num_batches = 0

    for batch_idx, (real_images, _) in enumerate(train_loader):
      batch_size = real_images.shape[0]
      # Flatten real images and move to device
      real_images = real_images.view(batch_size, -1).to(device)

      # Create real and fake labels
      real_labels = torch.ones(batch_size, 1).to(device)
      fake_labels = torch.zeros(batch_size, 1).to(device)

      ########### Train Discriminator ###########
      disc_optimizer.zero_grad() # Zero gradients for discriminator

      # Train with real images
      output_real = discriminator(real_images)
      disc_real_loss = criterion(output_real, real_labels)

      # Train with fake images
      z = torch.randn(batch_size, NOISE_DIM).to(device) # Generate noise
      fake_images = generator(z).detach() # Generate fake images and detach from generator's graph
      output_fake = discriminator(fake_images)
      disc_fake_loss = criterion(output_fake, fake_labels)

      # Total discriminator loss
      disc_loss = disc_real_loss + disc_fake_loss
      
      disc_loss.backward() # Backpropagate
      disc_optimizer.step() # Update discriminator weights


      ########### Train Generator ###########
      gen_optimizer.zero_grad() # Zero gradients for generator

      z = torch.randn(batch_size, NOISE_DIM).to(device) # Generate new noise for generator
      fake_images = generator(z) # Generate fake images (connected to generator's graph)
      output_gen = discriminator(fake_images) # Discriminator's output on fake images

      # Generator wants discriminator to classify fakes as real
      gen_loss = criterion(output_gen, real_labels)

      gen_loss.backward() # Backpropagate
      gen_optimizer.step() # Update generator weights


      ########### Mode Parameter Specific Training ###########
      if mode == "five_gen_one_disc":
        for _ in range(4): # 4 additional generator updates (total 5 G updates per 1 D update)
          gen_optimizer.zero_grad()
          z = torch.randn(batch_size, NOISE_DIM).to(device)
          fake_images_extra = generator(z)
          output_gen_extra = discriminator(fake_images_extra)
          gen_loss_extra = criterion(output_gen_extra, real_labels)
          gen_loss_extra.backward()
          gen_optimizer.step()
          gen_loss = gen_loss_extra # Update gen_loss for logging purposes with the last extra update

      if mode == "five_disc_one_gen":
        for _ in range(4): # 4 additional discriminator updates (total 5 D updates per 1 G update)
          disc_optimizer.zero_grad()
          
          # Re-evaluate disc_real_loss to ensure fresh gradients for real part
          output_real_extra = discriminator(real_images)
          disc_real_loss_extra = criterion(output_real_extra, real_labels)

          z = torch.randn(batch_size, NOISE_DIM).to(device)
          fake_images_extra = generator(z).detach()
          output_fake_extra = discriminator(fake_images_extra)
          disc_fake_loss_extra = criterion(output_fake_extra, fake_labels)

          disc_loss_extra = disc_real_loss_extra + disc_fake_loss_extra
          disc_loss_extra.backward()
          disc_optimizer.step()
          disc_loss = disc_loss_extra # Update disc_loss for logging purposes with the last extra update

      epoch_disc_loss += disc_loss.item()
      epoch_gen_loss += gen_loss.item()
      num_batches += 1

      # Optional: Print progress within epoch
      if (batch_idx + 1) % 100 == 0: # Print every 100 batches
          print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                f'D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}')

    # Average losses for the epoch
    avg_disc_loss = epoch_disc_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    history_disc_losses.append(avg_disc_loss)
    history_gen_losses.append(avg_gen_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Avg D Loss: {avg_disc_loss:.4f}, Avg G Loss: {avg_gen_loss:.4f}')
    
    # Save generated images periodically
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs: # Save at intervals and at the end
        print(f'--- Saving generated images for Epoch {epoch+1} ---')
        save_generated_images(epoch+1, generator, fixed_noise, results_dir=RESULTS_DIR)
        
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'generator_epoch_{epoch+1:03d}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINTS_DIR, f'discriminator_epoch_{epoch+1:03d}.pth'))
        print(f"Models saved at epoch {epoch+1}")

  print("Training finished.")
  # Save final models
  torch.save(generator.state_dict(), os.path.join(CHECKPOINTS_DIR, 'generator_final.pth'))
  torch.save(discriminator.state_dict(), os.path.join(CHECKPOINTS_DIR, 'discriminator_final.pth'))
  print("Final models saved.")

  return history_disc_losses, history_gen_losses

if __name__ == "__main__":
    # If train.py is run directly, use default parameters
    run_training(
        num_epochs=200,
        batch_size=128,
        mode="one_one",
        learning_rate=0.000086,
        weight_decay=0.00001
    )
