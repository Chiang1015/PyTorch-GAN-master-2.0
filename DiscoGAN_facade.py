import torch
import argparse
import itertools
import glob
import os
import datetime
import time

import torch.nn as nn
import numpy as np

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

input_shape = (opt.channels, opt.img_height, opt.img_width)
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

def sample_images(imgtag):
    imgs = next(iter(val_dataloader)) #batch=16
    G_AB.eval()
    G_BA.eval()
    with torch.inference_mode():
        real_A = imgs["A"].to(dtype=torch.float32,device=device)
        real_B = imgs["B"].to(dtype=torch.float32,device=device)

        fake_B = G_AB(real_A)  
        fake_A = G_BA(real_B)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)

    save_image(img_sample, f"./samples/DiscoGAN_{imgtag}.png", nrow=16, normalize=True)
    G_AB.train()
    G_BA.train()

#(in_size, out_size, normalize=True, dropout=0.0)
# into half size
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#(in_size, out_size, dropout=0.0)
#into double (Resnet)
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1), nn.InstanceNorm2d(out_size), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

#input:(batch,3,64,64) => img(batch,3,64,64)
class GeneratorUNet(nn.Module):
    def __init__(self, input_shape):
        super(GeneratorUNet, self).__init__()
        channels, _, _ = input_shape
        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(128, channels, 4, padding=1), nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x) #(batch,64,32,32)
        d2 = self.down2(d1) #(batch,128,16,16)
        d3 = self.down3(d2) #(batch,256,8,8)
        d4 = self.down4(d3) #(batch,512,4,4)
        d5 = self.down5(d4) #(batch,512,2,2)
        d6 = self.down6(d5) #(batch,512,1,1)

        u1 = self.up1(d6, d5) #(batch,1024,2,2)
        u2 = self.up2(u1, d4) #(batch,1024,4,4)
        u3 = self.up3(u2, d3) #(batch,512,8,8)
        u4 = self.up4(u3, d2) #(batch,256,16,16)
        u5 = self.up5(u4, d1) #(batch,128,32,32)

        return self.final(u5) #(batch,3,64,64)
    
#input:(batch,3,64,64) => val(batch,1,8,8)
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)
        #(1,8,8)
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256), #(batch,256,8,8)
            nn.ZeroPad2d((1, 0, 1, 0)), #(batch,256,9,9)
            nn.Conv2d(256, 1, 4, padding=1) #(batch,1,8,8)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
    

#object
G_AB = GeneratorUNet(input_shape).to(device)
G_BA = GeneratorUNet(input_shape).to(device)
D_A = Discriminator(input_shape).to(device)
D_B = Discriminator(input_shape).to(device)

#Loss
adversarial_loss = torch.nn.MSELoss().to(device)
cycle_loss = torch.nn.L1Loss().to(device)
pixelwise_loss = torch.nn.L1Loss().to(device)

#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0)
val_dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=16,
    shuffle=True,
    num_workers=0)

# Initialize weights
G_AB.apply(custom_weights_init)
G_BA.apply(custom_weights_init)
D_A.apply(custom_weights_init)
D_B.apply(custom_weights_init)

for epoch in range(opt.epoch, opt.n_epochs):
    prev_time = time.time() 
    for i, batch in enumerate(dataloader):         

         real_A = batch['A'].to(dtype=torch.float32,device=device)
         real_B = batch['B'].to(dtype=torch.float32,device=device)

         valid = torch.ones((real_A.size(0),1,8,8),dtype = torch.float32 , device=device ,requires_grad=False)
         fake =  torch.zeros((real_A.size(0),1,8,8),dtype = torch.float32 , device=device ,requires_grad=False)
         # ------------------
         #  Train Generators
         # ------------------
         G_AB.train()
         G_BA.train()
         optimizer_G.zero_grad()

         fake_B = G_AB(real_A)
         loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)

         fake_A = G_BA(real_B)
         loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)
         loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
         # Pixelwise translation loss
         loss_pixelwise = (pixelwise_loss(fake_A, real_A) + pixelwise_loss(fake_B, real_B)) / 2
         # Cycle loss
         loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
         loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
         loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

         # Total loss
         loss_G = loss_GAN + loss_cycle + loss_pixelwise

         loss_G.backward()
         optimizer_G.step()

         # -----------------------
         #  Train Discriminator A
         # -----------------------
         optimizer_D_A.zero_grad()
         # Real loss
         loss_real = adversarial_loss(D_A(real_A), valid)
         # Fake loss (on batch of previously generated samples)
         loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
         # Total loss
         loss_D_A = (loss_real + loss_fake) / 2

         loss_D_A.backward()
         optimizer_D_A.step()

         # -----------------------
         #  Train Discriminator B
         # -----------------------
         optimizer_D_B.zero_grad()
         # Real loss
         loss_real = adversarial_loss(D_B(real_B), valid)
         # Fake loss (on batch of previously generated samples)
         loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
         # Total loss
         loss_D_B = (loss_real + loss_fake) / 2

         loss_D_B.backward()
         optimizer_D_B.step()

         loss_D = 0.5 * (loss_D_A + loss_D_B) #for recode only
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {loss_D.item():.4f}][G loss: {loss_G.item():.4f}]",end='')
    print(f"[GAN loss: {loss_GAN.item():.4f}][CYC loss: {loss_cycle.item():.4f}][IDE loss: {loss_pixelwise.item():.4f}]--{time1round}")

    if epoch %100 == 0 or epoch ==499:
        sample_images(imgtag=epoch)
