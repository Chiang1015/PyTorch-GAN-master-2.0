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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_shape = (opt.channels, opt.img_size, opt.img_size)

def sample_image(n_row, imgtag):
    generator.eval()
    with torch.inference_mode():
        # Sample noise
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=False) 

        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)]) 
        labels = torch.tensor(labels, dtype=torch.long, device=device) #100*1
        gen_imgs = generator(z, labels)
    generator.train()
    save_image(gen_imgs, f"./samples/CGAN_{imgtag}.png", nrow=n_row, normalize=True)

#input:  noise(b,100)/ label(b,1) =>img(b,1,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)#(詞袋,維度)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),#1024 => 1*32*32
            nn.Tanh()
        )

    def forward(self, noise, labels):
        #拼接label(batch,10)與noise(batch,100) => (batch,110)
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        # img.view(64,1,32,32)
        return img

#input:(b,1,32,32)+(b,1)=>val(b,1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),#10+1*32*32 => 512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        #拼接img(batch,1*32*32)與label(batch,10) => (batch,1*32*32+10)
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)

        return validity


#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss functions
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure data loader
transform_seq = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
train_mnist_data = datasets.MNIST("./Resorces", train=False, download=False, transform=transform_seq)
dataloader = DataLoader(train_mnist_data, batch_size=opt.batch_size, shuffle=True)

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i , (img, labels) in enumerate(dataloader):
        batch_size = img.shape[0]

        valid = torch.ones((batch_size, 1), dtype=torch.float32, device=device, requires_grad=False)
        fake = torch.zeros((batch_size, 1), dtype=torch.float32, device=device, requires_grad=False)

        real_imgs = img.to(dtype=torch.float32,device=device)
        labels = labels.to(dtype=torch.long, device=device)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=True)
        gen_labels = torch.tensor(np.random.randint(0, opt.n_classes, batch_size), dtype=torch.long, device=device)

        gen_imgs = generator(z, gen_labels)
        validity = discriminator(gen_imgs, gen_labels)

        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        validity_real = discriminator(real_imgs, labels)
        d_loss_real = adversarial_loss(validity_real,valid)

        validity_fake = discriminator(gen_imgs.detach(),gen_labels)
        d_loss_fake = adversarial_loss(validity_fake, fake)

        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch == 199:
        sample_image(n_row=10, imgtag=epoch)   