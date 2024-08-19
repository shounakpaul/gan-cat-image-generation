import torch
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_size, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, ngf * 8, 4, 1,
                               0, bias=False),  # 1x1 -> 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1,
                               bias=False),  # 4x4 -> 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1,
                               bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1,
                               bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()  # Output: 64x64xnc
        )

    def forward(self, input):
        return self.main(input)


MODELS_PATH = 'models'

latent_size = 128
ngf = 64
ndf = 64
nc = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator(latent_size, ngf, nc).to(device)


netG.load_state_dict(torch.load(f'{MODELS_PATH}/generator.pth'))

# Set the model to evaluation mode
netG.eval()

# Generate noise
latent_size = 128  # Ensure this matches the latent size used during training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Adjust batch size as needed
fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)

# Generate images
with torch.no_grad():
    fake_images = netG(fixed_noise).detach().cpu()

# Visualize the images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(vutils.make_grid(fake_images, padding=2,
           normalize=True).permute(1, 2, 0))
plt.show()
