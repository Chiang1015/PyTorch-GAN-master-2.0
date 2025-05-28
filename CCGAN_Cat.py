#Conditional Contrastive GAN
#任務目標類似於：圖像修復（inpainting）、超解析度（super-resolution） 或 影像補全
import argparse
import torch
import glob
import os
import datetime
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from PIL import Image

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

input_shape = (opt.channels, opt.img_size, opt.img_size)

class ImageDataset(Dataset):
    def __init__(self, root, transforms_x=None, transforms_lr=None, mode='cat'):
        self.transform_x = transforms.Compose(transforms_x)
        self.transform_lr = transforms.Compose(transforms_lr)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #os.path.join() 會根據執行的系統，自動幫你選對的分隔符
        #glob.glob("資料夾/路徑/*.副檔名")
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        x = self.transform_x(img) #(batch,3,128,128)
        x_lr = self.transform_lr(img) #(batch,3,32,32)

        return {'x': x, 'x_lr': x_lr}

    def __len__(self):
        return len(self.files)

# Conv2d+BatchNorm2d+LeakyReLu+Dropout
#input:(b,in,a,b)=>(batch,out,a/2,b/2)
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        model = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            model.append(nn.BatchNorm2d(out_size, 0.8))
        model.append(nn.LeakyReLU(0.2))
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#input:(b,c,h,w)+(b,i,h/2,w/2)=>(batch,c+i,h/2,w/2)
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            #上採樣(可訓練參數)
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x, skip_input):
        x = self.model(x)
        out = torch.cat((x, skip_input), 1)
        return out
    
#input:(img,x_lr)=>(img)
#(b,3,128,128)+(b,3,32,32) => (b,3,128,128)
class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        channels, _, _ = input_shape
        #in=(batch,3,128,128)
        self.down1 = UNetDown(channels, 64, normalize=False)#(batch,64,64,64)
        self.down2 = UNetDown(64, 128)#(batch,128,32,32)
        self.down3 = UNetDown(128 + channels, 256, dropout=0.5)#(batch,256,16,16)
        self.down4 = UNetDown(256, 512, dropout=0.5)#(batch,512,8,8)
        self.down5 = UNetDown(512, 512, dropout=0.5)#(batch,512,4,4)
        self.down6 = UNetDown(512, 512, dropout=0.5)#(batch,512,2,2)
        #out=(batch,512,2,2)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256 + channels, 64)

        final = [nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, 1, 1), nn.Tanh()]
        self.final = nn.Sequential(*final)

    def forward(self, x, x_lr):
        #in=(batch,3,128,128)
        d1 = self.down1(x)#(batch,64,64,64)

        d2 = self.down2(d1)#(batch,128,32,32)
        d2 = torch.cat((d2, x_lr), 1)#+(batch,3,32,32)

        d3 = self.down3(d2) #(batch,256,16,16)
        d4 = self.down4(d3) #(batch,512,8,8)
        d5 = self.down5(d4) #(batch,512,4,4)
        d6 = self.down6(d5) #(batch,512,2,2)

        u1 = self.up1(d6, d5) #(batch,512+512,4,4)
        u2 = self.up2(u1, d4) #(batch,512+512,8,8)
        u3 = self.up3(u2, d3) #(batch,256+256,16,16)
        u4 = self.up4(u3, d2) #(batch,128+128+3,32,32)
        u5 = self.up5(u4, d1) #(batch,64+64,64,64)
        #final: (batch,128,64,64)=> (batch,128,128,128) => (batch,3,128,128)
        return self.final(u5) #(batch,3,128,128)

#input:img(b,3,128,128)=> val(b,1,16,16)
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape

        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(height / 2 ** 3), int(width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)#(1,16,16)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
                #每個通道獨立標準化成均值 0、方差 1
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        #( in,out,stride,Nor) => (batch,3,128,128)
        #(  3, 64,     2,  F) => (batch,64,64,64)
        #( 64,128,     2,  T) => (batch,128,32,32)
        #(128,256,     2,  T) => (batch,256,16,16)
        #(256,512,     1,  T) => (batch,512,16,16)

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        # => (b,1,16,16)

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
    
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

#img => masked_img
def apply_random_mask(imgs): # img_size =128 ,mask_size=32
    idx = np.random.randint(0, opt.img_size - opt.mask_size, (imgs.shape[0], 2))
    #idx(batch,2)
    masked_imgs = imgs.clone()
    for i, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size #右下定位點
        masked_imgs[i, :, y1:y2, x1:x2] = -1
        #[每張,全通道,左上,右下]

    return masked_imgs


def sample_image(saved_samples,imgtag):
    generator.eval()
    with torch.inference_mode(): #no_grad升級版
        # Generate inpainted image
        gen_imgs = generator(saved_samples["masked"], saved_samples["lowres"])
        # Save sample:沿y軸拼接
        sample = torch.cat((saved_samples["imgs"].data,saved_samples["masked"].data, gen_imgs.data, ), -2)
    generator.train()
    save_image(sample, f"./samples/CCGAN_{imgtag}.png" , nrow=5, normalize=True)
    #10組(遮,生,原)以2X5(nrow)儲存


#object
generator = Generator(input_shape).to(device)
discriminator = Discriminator(input_shape).to(device)
#loss
adversarial_loss = torch.nn.MSELoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)
#Data processor
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transforms_lr = [
    transforms.Resize((opt.img_size // 4, opt.img_size // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset('./Resorces', transforms_x=transforms_, transforms_lr=transforms_lr),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)


saved_samples = {}
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        imgs = batch["x"]
        imgs_lr = batch["x_lr"]
        masked_imgs = apply_random_mask(imgs)

        batch_size = imgs.shape[0]
        # Adversarial ground truths
        valid = torch.ones((batch_size, *discriminator.output_shape), dtype=torch.float32, device=device, requires_grad=False)
        fake = torch.zeros((batch_size, *discriminator.output_shape), dtype=torch.float32, device=device, requires_grad=False)


        real_imgs = imgs.to(dtype=torch.float32,device=device)
        imgs_lr = imgs_lr.to(dtype=torch.float32,device=device)
        masked_imgs = masked_imgs.to(dtype=torch.float32,device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(masked_imgs, imgs_lr)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]--{time1round}")

    # Save first ten samples:saved_samples is {dict.}
    if not saved_samples:
        saved_samples["imgs"] = real_imgs[:1].clone()
        saved_samples["masked"] = masked_imgs[:1].clone()
        saved_samples["lowres"] = imgs_lr[:1].clone()
    elif saved_samples["imgs"].size(0) < 10:
        saved_samples["imgs"] = torch.cat((saved_samples["imgs"], real_imgs[:1]), 0)#add to batch
        saved_samples["masked"] = torch.cat((saved_samples["masked"], masked_imgs[:1]), 0)
        saved_samples["lowres"] = torch.cat((saved_samples["lowres"], imgs_lr[:1]), 0)

    
    if (epoch-10) % 50 == 0 or epoch ==199:
        sample_image(saved_samples,epoch)