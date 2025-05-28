#Energy-Base GAN
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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
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
    z = torch.tensor(np.random.normal(0,1,(100,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
    generator.eval()
    with torch.inference_mode():
        gen_img = generator(z)
    save_image(gen_img,f'./samples/EBGAN_{imgtag}.png', nrow=10, normalize=True)
    generator.train

#鼓勵emb_imgs的多樣性
def pullaway_loss(embeddings): #embedding 一行視為一個向量
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True)) #std
    normalized_emb = embeddings / norm #normalize => 向量長度=1
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0)) 
    # M * Mt 向量之間內積，有batch^2項
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    #torch.sum(similarity)內積總和
    #減去對角線的1(自體內積)，僅在乎不同向量之間
    return loss_pt #

#input:z(b,62) => (b,1,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise) #(batch,128*8*8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img #(batch,1,32,32)
    
#input:(b,1,32,32) => (b,1,32,32) / (b,32)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img) #(batch,64,16,16)
        embedding = self.embedding(out.view(out.size(0), -1)) #(batch,32)
        out = self.fc(embedding) #(batch,64*16*16)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size)) #(batch,1,32,32)
        return out, embedding
    
#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#Loss
pixelwise_loss = nn.MSELoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader1 = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

lambda_pt = 0.05
margin = max(1, opt.batch_size / 64.0)
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader1):
        real_imgs = imgs.to(dtype=torch.float32,device=device)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0,1,(real_imgs.size(0),opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs = generator(z)
        recov_imgs, emb_imgs = discriminator(gen_imgs)

        g_loss = pixelwise_loss(recov_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(emb_imgs)
        #.detach使:z → gen_imgs → emb_imgs → pullaway_loss → backward → 更新 generator
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        recov_real, _ = discriminator(real_imgs)
        recov_fake, _ = discriminator(gen_imgs.detach())

        d_loss_real = pixelwise_loss(recov_real, real_imgs)
        d_loss_fake = pixelwise_loss(recov_fake, gen_imgs.detach())

        d_loss = d_loss_real
        # discriminator 太相信 generator（重建損失太小），會被懲罰
        # 不設定應該要判斷的結果，而設定不要的結果
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += margin - d_loss_fake

        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]--{time1round}")

    if epoch % 50 == 0 or epoch ==199:
        sample_image(imgtag=epoch)

