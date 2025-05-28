#pix2pix
import os
import time
import torch
import datetime
import argparse

import torch.nn as nn
import numpy as np
import glob

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        #Conv1d,Conv2d,Conv3d,ConvTranspose1d,ConvTranspose2d,ConvTranspose3d通用
        #如果是 Conv層，則權重(weight) 會初始化為均值 0、標準差 0.02 的常態分佈
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1: #1d/2d/3d通用
        #如果是 BatchNorm2d層，則權重(weight) 會初始化為均值 1、標準差 0.02 的常態分佈，偏置 (bias) 設為 0
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def sample_images(imgtag):
    generator.eval()
    with torch.inference_mode():
        imgs = next(iter(val_dataloader))
        real_A = imgs['A'].to(dtype=torch.float32,device=device)
        real_B = imgs['B'].to(dtype=torch.float32,device=device)
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, real_B.data, fake_B.data), -2)
    save_image(img_sample,f'./samples/pix2pix_{imgtag}.png',normalize=True)
    generator.train()

#input(b,in,h,w) => (b,out,h/2,w/2) 
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#input(b,c,h,w),(b,k,2h,2w) => (b,out+k,2h,2w)
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

#input:(b,3,256,256) => img(b,3,256,256)
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x) #(b,64,128,128)
        d2 = self.down2(d1)#(b,128,64,64)
        d3 = self.down3(d2)#(b,256,32,32)
        d4 = self.down4(d3)#(b,512,16,16)
        d5 = self.down5(d4)#(b,512,8,8)
        d6 = self.down6(d5)#(b,512,4,4)
        d7 = self.down7(d6)#(b,512,2,2)
        d8 = self.down8(d7)#(b,512,1,1)
        u1 = self.up1(d8, d7) #(b,1024,2,2)
        u2 = self.up2(u1, d6) #(b,1024,4,4)
        u3 = self.up3(u2, d5) #(b,1024,8,8)
        u4 = self.up4(u3, d4) #(b,1024,16,16)
        u5 = self.up5(u4, d3) #(b,512,32,32)
        u6 = self.up6(u5, d2) #(b,256,64,64)
        u7 = self.up7(u6, d1) #(b,128,128,128)

        return self.final(u7) #(b,3,256,256)
    
#input: (b,3,256,256),(b,3,256,256)=> val(b,1,16,16)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512), #(b,512,16,16)
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False) #(b,1,16,16)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}
    def __len__(self):
        return len(self.files)
    
#object
generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)
#loss
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixelwise = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,)
val_dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=0,)

patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4) #(1,16,16)
lambda_pixel = 100


for epoch in range(opt.epoch, opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(dtype=torch.float32,device=device)
        real_B = batch['B'].to(dtype=torch.float32,device=device)

        valid = torch.ones((real_A.size(0),*patch),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((real_A.size(0),*patch),dtype = torch.float32 , device=device ,requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)

        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

    time_dif = time.time()-prev_time
    time_1round = datetime.timedelta(seconds=time_dif)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {loss_D.item():.4f}][G loss: {loss_G.item():.4f}]",end='')
    print(f'[_pixel:{loss_pixel.item():.4f}][_gan:{loss_GAN.item():.4f}]--{time_1round}')

    if epoch %50==0 or epoch == 199:
        sample_images(imgtag=epoch)
