#Deep Regret AnalyticGAN
import torch
import argparse
import datetime
import time

import torch.autograd as autograd
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
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
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

#修正4:compute_gradient_penalty的WGAN-GP版本
def compute_gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10.0):
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
    #autograd.grad() 會傳回一個 tuple，裡面是每個 inputs 對應的梯度
    #這裡只有一個 inputs（interpolated），我們只取 [0]
    gradients = gradients.view(batch_size, -1) #(batch_size, num_features=32*32)
    gradient_norm = gradients.norm(2, dim=1)
    #L2 norm對dim=1 => (batch_size,) 代表梯度強度
    penalty =  ((gradient_norm - 1) ** 2).mean()
    #梯度強度與初始梯度1的差距取平均
    return penalty


def sample_images(imgtag):
    generator.eval()
    with torch.inference_mode():
        z = torch.tensor(np.random.normal(0,1,(10*10,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs = generator(z)
    save_image(gen_imgs,f"./samples/DraGAN_{imgtag}.png", nrow=10, normalize=True)
    generator.train()


#input:noise(batch,100) => img(batch,1,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4 #8
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
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

    def forward(self, noise): #(batch,100)
        out = self.l1(noise) #(batch,128*8*8)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) #(batch,128,8,8)
        img = self.conv_blocks(out) #(batch,1,32,32)
        return img

#input:(batch,1,32,32) => (batch,1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))#,nn.sigmoid()
        #修正1:刪除sigmoid()
    def forward(self, img):
        out = self.model(img) #(batch,128,2,2)
        out = out.view(out.shape[0], -1) #(batch,128*2*2)
        validity = self.adv_layer(out) #(batch,1)

        return validity

#object
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#Loss
adversarial_loss = torch.nn.BCELoss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

#init
generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)

# Loss weight for gradient penalty
lambda_gp = 10

for epoch in range(opt.n_epochs):
    prev_time = time.time() 
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.ones((imgs.size(0),1),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((imgs.size(0),1),dtype = torch.float32 , device=device ,requires_grad=False)
        real_imgs = imgs.to(dtype=torch.float32,device=device)
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0,1,(imgs.size(0),opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs = generator(z)
        #修正2:g_loss = -torch.mean(discriminator(gen_imgs))
        g_loss = -torch.mean(discriminator(gen_imgs))
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        #修正3:d_loss =-torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs.detach()))
        d_loss =-torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs.detach()))
        fake_imgs = generator(z).detach()
        #更穩定、收斂更平滑、模式崩潰更少
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs)
        d_loss += lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}][GP: {gradient_penalty.item():.4f}]--{time1round}")

    if epoch % 50 == 0 or epoch ==199:
        sample_images(imgtag=epoch)

    