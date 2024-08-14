from models.generator import Generator
from models.discriminator import Discriminator
from trainers.trainer import Trainer
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.utils as vutils
from config import OUTPUT_PATH, MODELS_PATH


class GAN:
    def __init__(self, latent_size, ngf, ndf, nc, lr, betas, device):
        self.latent_size = latent_size
        self.device = device
        self.generator = Generator(latent_size, ngf, nc).to(device)
        self.discriminator = Discriminator(ndf, nc).to(device)
        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas)
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)
        self.trainer = Trainer(self.discriminator, self.generator,
                               self.optimizerD, self.optimizerG, self.criterion, device)

    def train(self, epochs, dataloader):
        fixed_noise = torch.randn(
            64, self.latent_size, 1, 1, device=self.device)
        generator_losses = []
        discriminator_losses = []

        for epoch in range(epochs):
            for real_images, _ in dataloader:
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)
                lossD, real_score, fake_score = self.trainer.train_discriminator(
                    real_images, batch_size, self.latent_size)
                lossG = self.trainer.train_generator(
                    batch_size, self.latent_size)
            generator_losses.append(lossG)
            discriminator_losses.append(lossD)
            print(
                f'Epoch [{epoch+1}/{epochs}], Loss_D: {lossD:.4f}, Loss_G: {lossG:.4f}, Real_Score: {real_score:.4f}, Fake_Score: {fake_score:.4f}')
            with torch.no_grad():
                fake = self.generator(fixed_noise).detach().cpu()
                vutils.save_image(
                    fake, f'{OUTPUT_PATH}/generated_{epoch+1}.png', normalize=True)
        return generator_losses, discriminator_losses

    def save_models(self):
        torch.save(self.generator.state_dict(), f'{MODELS_PATH}/generator.pth')
        torch.save(self.discriminator.state_dict(),
                   f'{MODELS_PATH}/discriminator.pth')

    def load_models(self):
        self.generator.load_state_dict(
            torch.load(f'{MODELS_PATH}/generator.pth'))
        self.discriminator.load_state_dict(
            torch.load(f'{MODELS_PATH}/discriminator.pth'))
        self.generator.eval()
        self.discriminator.eval()
