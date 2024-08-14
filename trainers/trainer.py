import torch


class Trainer:
    def __init__(self, discriminator, generator, optimizerD, optimizerG, criterion, device):
        self.discriminator = discriminator
        self.generator = generator
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.criterion = criterion
        self.device = device

    def train_discriminator(self, real_images, batch_size, latent_size):
        self.discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_output = self.discriminator(real_images)
        real_loss = self.criterion(real_output, real_labels)
        real_loss.backward()
        real_score = real_output.mean().item()

        noise = torch.randn(batch_size, latent_size, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_output = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        fake_loss.backward()
        fake_score = fake_output.mean().item()

        self.optimizerD.step()
        lossD = real_loss + fake_loss
        return lossD.item(), real_score, fake_score

    def train_generator(self, batch_size, latent_size):
        self.generator.zero_grad()
        noise = torch.randn(batch_size, latent_size, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        labels = torch.ones(batch_size, 1, device=self.device)
        output = self.discriminator(fake_images)
        lossG = self.criterion(output, labels)
        lossG.backward()
        self.optimizerG.step()
        return lossG.item()
