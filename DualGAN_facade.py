import torch
import argparse
import itertools
import torch.nn as nn
import numpy as np
import glob
import os
import datetime
import time

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image

import torch.autograd as autograd

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)#(3,128,128)
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
    imgs = next(iter(val_dataloader)) #from val dataset(batch=)
    real_A = imgs['A'].to(dtype=torch.float32,device=device)
    real_B = imgs['B'].to(dtype=torch.float32,device=device)
    G_AB.eval()
    G_BA.eval()
    with torch.inference_mode():
       fake_B = G_AB(real_A)
       fake_A = G_BA(real_B)

       AB = torch.cat((real_A.data, fake_B.data), -2)  
       BA = torch.cat((real_B.data, fake_A.data), -2)
       img_sample = torch.cat((AB, BA), 0)
    
    save_image(img_sample, f"./samples/DualGAN_{imgtag}.png", nrow=16, normalize=True)
    G_BA.train()
    G_AB.train()

#懲罰不穩定的梯度
def compute_gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.size(0)
    device = real_data.device

    # 隨機係數 alpha 用來插值 real 和 fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    ones = torch.ones_like(d_interpolated)

    gradients = autograd.grad(
        outputs=d_interpolated, #微分的目標,如y(x)
        inputs=interpolated, #微分的變數,如dx,數量有32*32個
        grad_outputs=ones, #起始梯度
        create_graph=True, #保留計算圖,可以對這個梯度再做微分
        retain_graph=True, #保留整個計算圖
        only_inputs=True, #只針對 inputs 求梯度，而不考慮其他會牽涉的中間變數
    )[0]
    gradients = gradients.view(batch_size, -1) #(batch_size, num_features=32*32)
    gradient_norm = gradients.norm(2, dim=1)
    #L2 norm對dim=1 => (batch_size,) 代表梯度強度
    penalty =  ((gradient_norm - 1) ** 2).mean()
    #梯度強度與初始梯度1的差距取平均
    return penalty

#into half
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
#into double
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_size, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

#input:(batch,3,128,128) => (batch,3,128,128)
class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(nn.ConvTranspose2d(128, channels, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x) #(batch,64,64,64)
        d2 = self.down2(d1) #(batch,128,32,32)
        d3 = self.down3(d2) #(batch,256,16,16)
        d4 = self.down4(d3) #(batch,512,8,8)
        d5 = self.down5(d4) #(batch,512,4,4)
        d6 = self.down6(d5) #(batch,512,2,2)
        d7 = self.down7(d6) #(batch,512,1,1)

        u1 = self.up1(d7, d6) #(batch,1024,2,2)
        u2 = self.up2(u1, d5) #(batch,1024,4,4)
        u3 = self.up3(u2, d4) #(batch,1024,8,8)
        u4 = self.up4(u3, d3) #(batch,512,16,16)
        u5 = self.up5(u4, d2) #(batch,256,32,32)
        u6 = self.up6(u5, d1) #(batch,128,64,64)

        return self.final(u6) #(batch,3,128,128)
    
#input:(b,3,128,128)=>(b,1,14,14)
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discrimintor_block(in_features, out_features, normalize=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discrimintor_block(in_channels, 64, normalize=False),
            *discrimintor_block(64, 128),
            *discrimintor_block(128, 256),#(batch,256,16,16)
            nn.ZeroPad2d((1, 0, 1, 0)), #(batch,256,17,17)
            nn.Conv2d(256, 1, kernel_size=4) #(batch,1,14,14)
        )
    def forward(self, img):
        return self.model(img)
    

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

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
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)
#Loss
cycle_loss = torch.nn.L1Loss()
#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
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

# Loss weights
lambda_adv = 1
lambda_cycle = 10
lambda_gp = 10
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(dtype=torch.float32,device=device)
        real_B = batch['B'].to(dtype=torch.float32,device=device)
        # ----------------------
        #  Train Discriminators
        # ----------------------
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad() 
        fake_A = G_BA(real_B).detach()
        fake_B = G_AB(real_A).detach()

        # Domain A
        gp_A = compute_gradient_penalty(D_A, real_A.data, fake_A.data)
        D_A_loss = -torch.mean(D_A(real_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

        # Domain B
        gp_B = compute_gradient_penalty(D_B, real_B.data, fake_B.data)
        D_B_loss = -torch.mean(D_B(real_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

        # total loss
        D_loss = D_A_loss + D_B_loss

        D_loss.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        # Generator train less 
        if i % opt.n_critic == 0:
            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()
            fake_A = G_BA(real_B)
            fake_B = G_AB(real_A)
            # Reconstruct images
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)
            
            G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
            G_cycle = cycle_loss(recov_A, real_A) + cycle_loss(recov_B, real_B)
            # cycle *10 + adv *1 
            G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle
            G_loss.backward()
            optimizer_G.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    #G_adv: 異化能力 G_cycle:重構能力
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {D_loss.item():.4f}] [G loss: {G_loss.item():.4f}]",end='')
    print(f'[G_adv: {G_adv.item():.4f}] [G_cycle: {G_cycle.item():.4f}]--{time1round}')

    if epoch % 50 == 0 or epoch==199:
        sample_images(imgtag=epoch)
