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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    generator.eval()
    with torch.inference_mode():
        # Sample noise
        z = torch.randn((n_row**2, opt.latent_dim), device=device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)]) 
        labels = torch.tensor(labels, dtype=torch.long, device=device) #64*1
        gen_imgs = generator(z, labels)
    generator.train()
    save_image(gen_imgs, f"./samples/ACGAN_{imgtag}.png", nrow=n_row, normalize=True)

#input:noise(b,100) + label(b,1) =>img(b,1,32,32)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)#產生n_class X latent_dim的標籤，每組向量對應一種類別

        self.init_size = opt.img_size // 4  #32/4取商
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))#latent_dim=100, 128*8*8=8192

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),# (64,128,8,8)
            nn.Upsample(scale_factor=2), # (64,128,16,16)
            #(in_channels, out_channels, kernel_size, stride, padding)
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #(64,128,16,16)
            nn.BatchNorm2d(128, 0.8), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), #(64,128,32,32)
            nn.Conv2d(128, 64, 3, stride=1, padding=1), #(64,64,32,32)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), #(64,1,32,32)
            nn.Tanh(),
        )

    def forward(self, noise, labels): #noise=64*100, labels=64*1 but label_emb=64*100 
        gen_input = torch.mul(self.label_emb(labels), noise)#逐元素相乘
        out = self.l1(gen_input) #(64,100) => (64,8192)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size) #(64,128,8,8)
        img = self.conv_blocks(out) #(64,1,32,32)
        return img

#input: (b,1,32,32) => val(b,1) / label(b,10)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            #(in_channels, out_channels, kernel_size, stride, padding)
            #Dropout2d:會機率歸零一整個通道，而不是一個元素
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block #list
        # *表示解包list
        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False), #無BN #(64,1,32,32)=>(64,16,16,16) stride=2之故
            *discriminator_block(16, 32), #其他都有BN #(64,16,16,16)=>(64,32,8,8)
            *discriminator_block(32, 64),  #(64,32,8,8)=>(64,64,4,4)
            *discriminator_block(64, 128), #(64,64,4,4)=>(64,128,2,2)
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()) #機率
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax(dim=1))#分類

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1) #(64,128,2,2)=>(64,512)
        validity = self.adv_layer(out) #(64,512)=>(64,1)
        label = self.aux_layer(out) #(64,512)=>(64,10)

        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss().to(device)  #用於真假機率間
auxiliary_loss = torch.nn.CrossEntropyLoss().to(device) #用於分類標籤間

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# Initialize weights
generator.apply(custom_weights_init)
discriminator.apply(custom_weights_init)


transform_seq = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_mnist_data = datasets.MNIST("./Resorces", train=False, download=False, transform=transform_seq)
dataloader = DataLoader(dataset=train_mnist_data, batch_size = opt.batch_size, shuffle = True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(opt.n_epochs):
    prev_time = time.time()
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        fake = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        # Configure input
        real_imgs = imgs.to(dtype=torch.float32, device=device)
        labels = labels.to(dtype=torch.long, device=device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn((batch_size, opt.latent_dim), device=device)
        gen_labels = torch.tensor(np.random.randint(0, opt.n_classes, batch_size), dtype=torch.long, device=device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = 0.5 * (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) 

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = 0.5 * (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels))

        # Total discriminator loss
        d_loss = 0.5 * (d_real_loss + d_fake_loss) 

        # Calculate discriminator accuracy:
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0) #分類(10)標籤機率 在垂直方向上合併
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]--{time1round}")
    if epoch % 50 == 0 or epoch == 199:
        sample_image(n_row=10, imgtag=epoch)
