import argparse
import itertools
import torch
import datetime
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print('opt:',opt)#參數列印

img_shape = (opt.channels, opt.img_size, opt.img_size) # (1, 32, 32) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2).to(device)
    sampled_z = torch.tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim)),dtype=torch.float32).to(device) # (0,1,(batch=64, 10))
    z = sampled_z * std + mu
    return z

def sample_image(n_row, imgtag):
    """Saves a grid of generated digits"""
    # Sample noise
    decoder.eval()
    with torch.inference_mode():
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=False) 
        gen_imgs = decoder(z)
    decoder.train()
    save_image(gen_imgs, f"./samples/AAEGAN_{imgtag}.png", nrow=n_row, normalize=True)#路徑/檔名
    #合併100張圖變成1張大圖存起來

#input: (b,1,32,32) => (b,10)
class Encoder(nn.Module): #img(32X32) => latent space(seed)(10)
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512), #prob相乘以計算元素數量
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, opt.latent_dim) #latent_dim = 10
        self.logvar = nn.Linear(512, opt.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1) #flatten
        x = self.model(img_flat) 

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

#input: (b,10) => (b,1,32,32)
class Decoder(nn.Module): #latent space(seed) => img
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),#(batch,512,32*32)
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

#input:(b,10) =>val(b,1)
class Discriminator(nn.Module): #latent space(seed) => 0 or 1
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), #
            nn.Sigmoid(),#用leakyrelu不會有負值，所以用sigmoid，而softmax用於可能有負值
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


#object
encoder = Encoder().to(device)
decoder = Decoder().to(device)  
discriminator = Discriminator().to(device)

#loss
adversarial_loss = torch.nn.BCELoss().to(device)
pixelwise_loss = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
#合併 Encoder 和 Decoder 的參數，使它們能夠一起被優化
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_mnist_data = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq)
dataloader = DataLoader(dataset=train_mnist_data, batch_size = opt.batch_size, shuffle = True)

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = torch.ones((imgs.shape[0], 1), dtype=torch.float32, device=device)
        fake = torch.zeros((imgs.shape[0], 1), dtype=torch.float32, device=device)

        # Configure input
        real_imgs = imgs.to(dtype=torch.float32, device=device)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs) #seed type
        decoded_imgs = decoder(encoded_imgs) #img type

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)
        # 0.001*BCELoss(結果vs真標籤) + 0.999*L1Loss(生成圖vs真圖)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth 
        z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)),dtype=torch.float32, device=device, requires_grad=True).to(device)# (64, 10)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(encoded_imgs.detach()), valid)
        fake_loss = adversarial_loss(discriminator(z), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch==199:
        sample_image(n_row=10, imgtag=epoch)
    