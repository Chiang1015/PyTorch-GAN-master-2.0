import torch
import time
import argparse
import datetime

import torch.nn as nn
import numpy as np

from torchvision import datasets #現成的資料集庫
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

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
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_image(imgtag):
    row =10
    generator.eval()
    z = torch.randn((row**2, opt.latent_dim), device=device)
    with torch.inference_mode():
        sample = generator(z)
    save_image(sample,f'./samples/WGAN_div_{imgtag}.png', nrow=row, normalize=True)
    generator.train()

#input:(b,100)=> (b,1,28,28)
# linear only
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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img
    
#input:(b,1,28,28) => val(b,1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)


k = 10 # even
p = 2  # even
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.to(dtype=torch.float32,device=device).requires_grad_()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        z = torch.randn((imgs.size(0), opt.latent_dim), device=device)
        fake_imgs = generator(z).detach().requires_grad_() #!!

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)

        # val to loss:real_imgs
        real_grad_out = torch.ones_like(real_validity, device=device)
        real_grad = torch.autograd.grad(
            outputs=real_validity,
            inputs=real_imgs,
            grad_outputs=real_grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1).pow(p / 2)

        # val to loss:fake_imgs
        fake_grad_out = torch.ones_like(fake_validity, device=fake_validity.device)
        fake_grad = torch.autograd.grad(
            outputs=fake_validity,
            inputs=fake_imgs,
            grad_outputs=fake_grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1).pow(p / 2)

        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp
        d_loss.backward()
        optimizer_D.step()

        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            fake_imgs = generator(z)
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()
    time1round = datetime.timedelta(seconds = time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]",end='')
    print(f'--{time1round}')

    if epoch %50 ==0 or epoch == 199:
        sample_image(imgtag=epoch)
