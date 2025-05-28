import argparse
import torch
import datetime
import time

import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torchvision.transforms as transforms

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

def sample_image(n_row, imgtag):
    # Sample noise
    z = torch.tensor(np.random.normal(0,1,(n_row**2,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
    generator.eval()
    with torch.inference_mode():       
        gen_imgs = generator(z)
    generator.train()
    save_image(gen_imgs, f"./samples/BEGAN_{imgtag}.png", nrow=n_row, normalize=True)

#input:(b,62) =>(b,1,32,32) 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))#(64, 62)->(64, 128*8*8)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),#(64, 128, 8, 8)->(64, 128, 16, 16)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2), #(64, 128, 16, 16)->(64, 128, 32, 32)
            nn.Conv2d(128, 64, 3, stride=1, padding=1), #(64, 64, 32, 32)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), #(64, 1, 32, 32)
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise) #(64, 62) -> (64, 8192)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) #(64, 128, 8, 8)
        img = self.conv_blocks(out)
        return img

#輸入(64, 1, 32, 32)的圖片，輸出(64, 1, 32, 32)的圖片
#input: (b,1,32,32) => (b,1,32,32)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        #(in_channels, out_channels, kernel_size, stride=2, padding)
        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 3, 2, 1),
            nn.ReLU())#(64, 1, 32, 32)->(64, 64, 16, 16)
        # Fully-connected layers
        self.down_size = opt.img_size // 2 #16
        down_dim = 64 * (opt.img_size // 2) ** 2 
        self.fc = nn.Sequential(
            #瓶頸結構
            nn.Linear(down_dim, 32),#(64, 64*16*16)->(64, 32)
            #捨棄不重要的資訊
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),

            nn.Linear(32, down_dim),#(64, 32)->(64, 64*16*16)
            #從壓縮後的資訊重建輸入
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, opt.channels, 3, 1, 1))
        #(64, 64, 16, 16)->(64, 64, 32, 32)->(64, 1, 32, 32)
    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))#(64, 64, 16, 16)->(64, 64*16*16)
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        #up((64, 64, 16, 16))->(64, 1, 32, 32)
        return out

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# Initialize weights
generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

# Configure data loader
transform_seq = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])
train_mnist_data = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq)
dataloader = DataLoader(train_mnist_data, batch_size=opt.batch_size, shuffle=True) #pin_memory=True

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# BEGAN hyper parameters
gamma = 0.75 #平衡係數
lambda_k = 0.001 #k的學習率
k = 0.02 #0~1
for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        batch_size = imgs.shape[0]
        real_imgs = imgs.to(dtype=torch.float32,device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.tensor(np.random.normal(0,1,(batch_size,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs)) #重建誤差
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake #希望真實圖片前後完全相同，假圖片前後差異越大

        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights
        # ----------------
        diff = torch.mean(gamma * d_loss_real - d_loss_fake) #兩種重建損失的平衡
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # k與0取大,再與1取小

        #在d_loss_real下降而d_loss_fake上升的過程中,k會下降
        #k若下降,表示對假圖片更寬容(專注於讓真圖片前後相同),導致d_loss_fake下降
        #k又會因此上升,引發注意力轉向假圖片,導致d_loss_real上升,反覆循環       

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).item()
        #越小表示(真實圖片重建誤差越小)&(重建損失達到平衡)
        #穩定不變表示達到平衡

    time1round = datetime.timedelta(seconds=time.time()-prev_time)    
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}][M: {M:.4f}][k: {k:.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch == 199:
        sample_image(n_row=10, imgtag=epoch)
    