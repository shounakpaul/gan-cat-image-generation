import os
import shutil
from config import *
from data.data_pipeline import DataPipeline
from models.gan import GAN
from utils.plot_loss import plot_loss


def main():
    # Data pipeline
    print("Initializing data pipeline...")
    data_pipeline = DataPipeline(DATASET_PATH, IMAGE_SIZE, BATCH_SIZE)
    dataloader = data_pipeline.get_dataloader()
    print(f"Loaded dataset with {len(dataloader.dataset)} images.")

    # Initialize GAN
    print("Initializing GAN model...")
    gan = GAN(LATENT_SIZE, NGF, NDF, NC, LEARNING_RATE, BETAS, DEVICE)
    print("GAN model initialized.")

    # Train GAN
    print(f"Starting training for {EPOCHS} epochs...")

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    if os.path.exists(MODELS_PATH):
        shutil.rmtree(MODELS_PATH)
    os.makedirs(MODELS_PATH, exist_ok=True)

    generator_losses, discriminator_losses = gan.train(EPOCHS, dataloader)
    print("Training completed.")

    # Save models
    print("Saving trained models...")
    gan.save_models()
    print(f"Models saved to {MODELS_PATH}.")

    # Visualize losses
    print("Visualizing and saving loss plots...")
    plot_loss(generator_losses, discriminator_losses,
              f'{OUTPUT_PATH}/loss_plot.png')
    print(f"Loss plot saved to {OUTPUT_PATH}/loss_plot.png.")


if __name__ == "__main__":
    print("Starting GAN training script...")
    main()
    print("Script finished successfully.")
