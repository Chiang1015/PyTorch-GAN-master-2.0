#Deep Convolutional GAN
import torch
import torch
import argparse
import datetime
import time

import torch.nn as nn
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size) #(1,32,32)
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

def sample_images(row,imgtag):
    generator.eval()
    with torch.inference_mode():
        seed = torch.tensor(np.random.normal(0,1,(row**2,opt.latent_dim)) , dtype=torch.float32, device=device)
        sample_img = generator(seed)
    generator.train()
    save_image(sample_img, f'./samples/DCGAN_{imgtag}.png', nrow=row , normalize = True)

#input:noise(b,100) =>img(b,1,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4 #8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),#(batch,128,16,16)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),#(batch,128,16,16)

            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),#(batch,128,32,32)
            nn.Conv2d(128, 64, 3, stride=1, padding=1),#(batch,64,32,32)

            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),#(batch,1,32,32)
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
#input:(b,1,32,32) => val(b,1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(#(batch,1,32,32)
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),#(batch,128,2,2)
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4 #2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

#object    
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#loss
adversarial_loss = torch.nn.BCELoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#Initialize weights
generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

for epoch in range(opt.n_epochs):
    prev_time = time.time() 
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        valid = torch.ones((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)

        real_imgs = imgs.to(dtype=torch.float32,device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0,1,(batch_size,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch ==199:
        sample_images(row= 10, imgtag= epoch)   
