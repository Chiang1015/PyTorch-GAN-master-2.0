import argparse
import torch
import datetime
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size) # (1, 28, 28)

def sample_image(n_row, imgtag):
    """Saves a grid of generated digits"""
    # Sample noise
    generator.eval()
    with torch.inference_mode():
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs = generator(z)
    generator.train()
    save_image(gen_imgs, f"./samples/BGAN_{imgtag}.png", nrow=n_row, normalize=True)

#input:(b, 100) -> img(b, 1, 28, 28)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), #(1024) -> (28*28)
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

#input:(b, 1, 28, 28) -> val(b, 1)  
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

def boundary_seeking_loss(y_pred):
    eps = 1e-8  # 避免 log(0)
    return 0.5 * torch.mean((torch.log(y_pred + eps) - torch.log(1 - y_pred + eps)) ** 2)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

discriminator_loss = torch.nn.BCELoss().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_mnist_data = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq)
dataloader = DataLoader(dataset=train_mnist_data, batch_size = opt.batch_size, shuffle = True)

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.ones((imgs.shape[0], 1), dtype=torch.float32, device=device)
        fake = torch.zeros((imgs.shape[0], 1), dtype=torch.float32, device=device)

        real_imgs = imgs.to(device)

        # -----------------
        #  Train Generator
        # ----------------- 
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)),dtype=torch.float32, device=device, requires_grad=True).to(device)
        gen_imgs = generator(z)
        g_loss = boundary_seeking_loss(discriminator(gen_imgs))
        # 不是最大化dis(gen_imgs)而是使其趨近0.5
        # 意味著判別器模糊的狀態，從而逼近數據邊界而不是模仿真實數據
        # 可以保留多樣化，而不是某些特定模式
        # 但是導致梯度很小，使得生成器的學習速度變慢且可能產生模糊的結果
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = discriminator_loss(discriminator(real_imgs), valid)
        fake_loss = discriminator_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch ==199:
        sample_image(n_row=10, imgtag=epoch)
