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
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)#(1,28,28)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def custom_weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d,nn.BatchNorm1d,nn.BatchNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def sample_image(imgtag):
    row = 10
    z = torch.empty((row**2,opt.latent_dim),device=device).uniform_(-1,1)
    generator.eval()
    with torch.inference_mode():
        sample = generator(z)
    save_image(sample,f'./samples/softmaxGAN_{imgtag}.png', nrow=row, normalize=True)
    generator.train()

#input:(b,100) => img(b,1,32,32)
#linear only
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
    
#input:img(b,1,32,32) => val(b,1)
#linear only
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.img_size ** 2, 512),
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
#loss
adversarial_loss = torch.nn.BCELoss()
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader1 = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader1):
        batch_size = imgs.size(0)
        real_imgs = imgs.to(dtype=torch.float32,device=device)
        z = torch.empty((batch_size, opt.latent_dim), device=device,requires_grad=False).uniform_(-1, 1)

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        # Adversarial ground truths
        g_target = 1 / (batch_size * 2) #1/64*64
        d_target = 1 / batch_size #1/64

        gen_imgs = generator(z)
        #Loss
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs)
        sum_exp_D = torch.sum(torch.exp(-d_real)) + torch.sum(torch.exp(-d_fake))
        
        d_loss = d_target * torch.sum(d_real) + torch.log(sum_exp_D+ 1e-8 )
        d_loss.backward(retain_graph=True)
        optimizer_D.step()

        # loss of generator
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs)
        sum_exp_G = torch.sum(torch.exp(-d_real)) + torch.sum(torch.exp(-d_fake))

        g_loss = g_target * (torch.sum(d_real) + torch.sum(d_fake)) + torch.log(sum_exp_G+ 1e-8 )
        g_loss.backward()
        optimizer_G.step()

    time1round = datetime.timedelta(seconds=time.time() - prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]",end='')
    print(f'--{time1round}')

    if epoch % 50 == 0 or epoch == 199:
        sample_image(imgtag=epoch)
