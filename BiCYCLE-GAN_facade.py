import argparse
import torch
import glob
import os
import datetime
import time

from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset
from torchvision.models import resnet18
from PIL import Image

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = (opt.channels, opt.img_height, opt.img_width)

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

#**這裡採用cLR模式生成,因為與encoder訓練區採用cLR素材(fake_B1,RandomZ)有關
#如果encoder採用VAE素材(fake_B,Z)，這裡要改成先用img_B=>mu,logvar=>z,再與real_A進行生成
#但是VAE模式不含隨機噪聲，所以更像原圖的重建
def sample_images(imgtag):
    """Saves a generated sample from the validation set"""
    
    imgs = next(iter(val_dataloader))#8張圖
    img_samples = None
    generator.eval()
    with torch.inference_mode():
        for img_A, img_B in zip(imgs["A"], imgs["B"]):#跑8圈,每輪使用一張原圖+8張生成，向下堆積
            # Repeat input image by number of desired columns
            real_A = img_A.view(1, *img_A.shape).repeat(opt.latent_dim, 1, 1, 1)
            #將(3,128,128)=>(1,3,128,128)=>複製latent_dim次(8,3,128,128)

            real_A = real_A.to(dtype=torch.float32,device=device)

            # Sample latent representations
            sampled_z =torch.tensor(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=False)
            #(8,8)

            # Generate samples 
            fake_B = generator(real_A, sampled_z)

            # Concatenate samples horisontally
            fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)#for遍歷第一個維度batch=8
            #(8,3,128,128)=>(3,128,128*8)
            img_sample = torch.cat((img_A, fake_B), -1)#加上img_A於前=>(3,128,128*9)
            img_sample = img_sample.view(1, *img_sample.shape)#save_image要求 (batch, channels, height, width)
            #(1,3,128,128)
            # Concatenate with previous samples vertically
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
            #使圖片有不同real_A的變化(向下)，隨機的變化(向右)

    save_image(img_samples, f"./samples/BicycleGAN_{imgtag}.png", normalize=True)
    generator.train()

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z =torch.tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=False)
    z = sampled_z * std + mu
    return z

#input:(b,in,h,w) => (b,out,h/2,w/2)
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#
#input:(b,c,h,w) => (b,c+s,2h,2w)
class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x
    
#input:(b,3,128,128)+(batch,8) => (b,3,128,128)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape #3,128,128

        self.fc = nn.Linear(latent_dim, self.h * self.w)#(8,128*128)

        self.down1 = UNetDown(channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):       
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        #z(batch,8)=>z(batch,1,128,128)
        #x(batch,3,128,128)
        d1 = self.down1(torch.cat((x, z), 1))#(batch,4,128,128)=>d1(batch,64,64,64)

        d2 = self.down2(d1)#d2(batch,128,32,32)
        d3 = self.down3(d2)#d3(batch,256,16,16)
        d4 = self.down4(d3)#d4(batch,512,8,8)
        d5 = self.down5(d4)#d5(batch,512,4,4)
        d6 = self.down6(d5)#d6(batch,512,2,2)
        d7 = self.down7(d6)#d7(batch,512,1,1)

        u1 = self.up1(d7, d6)#u1(batch,512+512,2,2)
        u2 = self.up2(u1, d5)#u2(batch,512+512,4,4)
        u3 = self.up3(u2, d4)#u3(batch,512+512,8,8)
        u4 = self.up4(u3, d3)#u4(batch,256+256,16,16)
        u5 = self.up5(u4, d2)#u5(batch,128+128,32,32)
        u6 = self.up6(u5, d1)#u6(batch,64+64,64,64)

        return self.final(u6) #(batch,3,128,128)

#input:(b,3,128,128) => mu(b,8), logvar(b,8)
#Resnet18
class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Encoder, self).__init__()
        resnet18_model = resnet18(weights=None)
        #resnet僅接受(batch,3,?,?)圖
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        #list(resnet18_model.children())=>[conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        #[:-3]移除後三層 =>使其僅留下提取特徵能力=>輸出輸出中間特徵 (b, 256, 8, 8)

        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        #(batch_size, 256, 1, 1)平均池化保留較多全局資訊
        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out) #(b,256,1,1)
        out = out.view(out.size(0), -1) #(b,256)

        mu = self.fc_mu(out) #(b,8)
        logvar = self.fc_logvar(out)#(b,8)
        return mu, logvar

#input:(b,3,128,128) => output=[(b,1,8,8),(b,1,4,4),(b,1,2,2)]
#建立compute_loss(x,gt)
class MultiDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        channels, _, _ = input_shape #3,128,128
        # Extracts discriminator models
        self.models = nn.ModuleList()#儲存模組的專門列表
        for i in range(3):#建立3個，命名為disc_0,disc_1,disc_2，命名為disc_0,disc_1,disc_2
            #根據forward:
            #disc_0的輸入(batch,3,128,128)=>輸出(batch,1,8,8)
            #disc_1的輸入(batch,3,64,64)=>輸出(batch,1,4,4)
            #disc_2的輸入(batch,3,32,32)=>輸出(batch,1,2,2)
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),#(batch,512,8,8)
                    nn.Conv2d(512, 1, kernel_size=3, padding=1)#(batch,1,8,8)
                    #(in,out,ker,stride=1,padding)
                ),
            )

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False)
        #(kernel_size=2, stride=2, padding=0, count_include_pad=False)
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss
    
    #多尺度學習
    def forward(self, x):
        outputs = []
        for m in self.models:#需要nn.ModuleList()
            outputs.append(m(x))
            x = self.downsample(x)

        #output=[(batch,1,8,8),(batch,1,4,4),(batch,1,2,2)]       
        return outputs

#初始化dataset=ImageDataset(root,input_shape) 
#使用Dataloader(dataset,batch,shuffle)自動調用getitem
class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),#resize(128,128,雙三插值)
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #路徑為root+mode
        #"*.*" 表示所有副檔名的檔案
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])#從files裡循環讀取圖片
        w, h = img.size
        #crop(左上點座標,右下點座標)
        img_A = img.crop((0, 0, w / 2, h))#左半圖片
        img_B = img.crop((w / 2, 0, w, h))#右半圖片

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB") #(H, W, C)
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB") #w隨機反序(水平反轉)

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
    
#object
generator = Generator(opt.latent_dim, input_shape).to(device)
encoder = Encoder(opt.latent_dim, input_shape).to(device)
D_VAE = MultiDiscriminator(input_shape).to(device)
D_LR = MultiDiscriminator(input_shape).to(device)
#Loss
mae_loss = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#initial weight
generator.apply(custom_weights_init)
D_VAE.apply(custom_weights_init)
D_LR.apply(custom_weights_init)
#encoder使用resnet18不用初始化

dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, input_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    ImageDataset("./Resorces/%s" % opt.dataset_name, input_shape, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)


valid = 1
fake = 0
for epoch in range(opt.epoch, opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(dtype=torch.float32,device=device)
        real_B = batch['B'].to(dtype=torch.float32,device=device)

        #  Train Generator and Encoder
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        # ----------
        # cVAE-GAN
        # ----------
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)
        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)

        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # 如果 mu 太大，表示學到的分佈偏離 0，KL 散度會變大，懲罰這種偏離。
        # 如果 logvar 太小（表示標準差接近 0），模型會變成單點估計，失去 VAE 的「隨機性」，KL 散度會變大來懲罰它。
        # 這個損失的作用是讓 mu 靠近 0，sigma 靠近 1，使得學到的潛在變數近似於標準正態分佈。

        # ----------
        # cLR-GAN
        # ----------
        sampled_z = torch.tensor(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim)), dtype=torch.float32, device=device, requires_grad=True)
        fake_B1 =  generator(real_A, sampled_z)
        loss_LR_GAN = D_LR.compute_loss(fake_B1, valid)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl
        # lambda_pixel=10 強調像素相似性，讓圖像更清晰、與真實圖像匹配
        # lambda_kl=0.01 適當增加潛在變數的多樣性，讓 BicycleGAN 生成不同變化的結果
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # ---------------------
        # Train Encoder
        # ---------------------
        mu1, _ = encoder(fake_B1)
        loss_latent = opt.lambda_latent * mae_loss(mu1, sampled_z)
        #lambda_latent = 0.5
        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------
        optimizer_D_VAE.zero_grad()
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)
        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------
        optimizer_D_LR.zero_grad()
        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(fake_B1.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D VAE_loss: {loss_D_VAE.item():.4f}, LR_loss:{loss_D_LR.item():.4f}, pixel:{loss_pixel.item():.4f}, kl:{loss_kl.item():.4f}][G loss: {loss_GE.item():.4f}][Encoder loss:{loss_latent.item():.4f}]",end='')
    print(f'--{time1round}')
    
    if epoch % 50 == 0 or epoch ==199:
        sample_images(imgtag=epoch)

