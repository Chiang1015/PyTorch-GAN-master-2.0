import os
import gzip
import pickle
import torch
import torch
import argparse
import datetime
import time

import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

from PIL import Image
from torchvision import datasets
from urllib.request import urlopen
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets

# Downloader:MNISTM
# MNISTM (root, mnist_root="data", train=True, transform=None, target_transform=None, download=False)
class MNISTM(data.Dataset):
    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"
    raw_file = "keras_mnistm.pkl"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        data_file = self.training_file if self.train else self.test_file
        self.data, self.targets = torch.load(os.path.join(self.root, data_file),weights_only=True)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return all(os.path.exists(os.path.join(self.root, fname)) for fname in [self.training_file, self.test_file])

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        print("Downloading:", self.url)
        raw_path = os.path.join(self.root, self.raw_file)
        with open(raw_path, "wb") as f:
            f.write(urlopen(self.url).read())

        print("Extracting...")
        with gzip.open(raw_path, "rb") as zip_f:
            with open(raw_path[:-3], "wb") as out_f:
                out_f.write(zip_f.read())
        os.remove(raw_path)

        print("Processing...")
        with open(raw_path[:-3], "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")

        train_data = torch.ByteTensor(mnist_m_data[b"train"])
        test_data = torch.ByteTensor(mnist_m_data[b"test"])

        train_labels = datasets.MNIST(self.mnist_root, train=True, download=True).targets
        test_labels = datasets.MNIST(self.mnist_root, train=False, download=True).targets

        torch.save((train_data, train_labels), os.path.join(self.root, self.training_file))
        torch.save((test_data, test_labels), os.path.join(self.root, self.test_file))

        print("Done!")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size) #(3,32,32)
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
    couple_number = int(row**2/2)#50組
    coupled_generators.eval()
    with torch.inference_mode():
        seed = torch.tensor(np.random.normal(0,1,(couple_number,opt.latent_dim)), dtype=torch.float32, device=device)
        sample_img1, sample_img2 = coupled_generators(seed)
        couple_img = torch.cat([sample_img1,sample_img2],dim=0)       
    coupled_generators.train()
    save_image(couple_img, f'./samples/COGAN_{imgtag}.png',nrow=row, normalize=True)

#input:noise(b,100) => img1 /img2(b,3,32,32)
class CoupledGenerators(nn.Module):
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        self.init_size = opt.img_size // 4 #8
        self.fc = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.Sequential(#(batch,128,8,8)
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),#(batch,128,32,32)
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)#(batch,128,32,32)

        img1 = self.G1(img_emb)#(batch,3,32,32)
        img2 = self.G2(img_emb)#(batch,3,32,32)
        return img1, img2
    
#input:img1/img2(b,3,32,32) => val_1/val_2(batch,1) no sig
class CoupledDiscriminators(nn.Module):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),#(batch,128,2,2)
        )
        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4 #2
        self.D1 = nn.Linear(128 * ds_size ** 2, 1)
        self.D2 = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        # img1: validity1
        out = self.shared_conv(img1)#(batch,128,2,2)
        out = out.view(out.shape[0], -1)#(batch,128*2*2)
        validity1 = self.D1(out)#(batch,1)

        # img2: validity2
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2

#object
# Initialize models
coupled_generators = CoupledGenerators().to(device)
coupled_discriminators = CoupledDiscriminators().to(device)

#Loss
adversarial_loss = torch.nn.MSELoss().to(device)

#optimizer
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Initialize weights
coupled_generators.apply(custom_weights_init)
coupled_discriminators.apply(custom_weights_init)

#dataset
# 1st: MNIST(channel = 1)
transform_seq1 = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) #(mean,std)單通道
dataset_1 = datasets.MNIST("./Resorces", train=True, download=False, transform=transform_seq1)
dataloader1 = DataLoader(dataset_1, batch_size=opt.batch_size, shuffle=True)

# 2ed: MNIST-M(channel = 3)
transform_seq2 = transforms.Compose([transforms.Resize(opt.img_size),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])#(mean,std)三通道
dataset_2 = MNISTM(root="./Resorces/mnist_m_data", train=True, transform=transform_seq2, download=False)
dataloader2 = DataLoader(dataset_2,batch_size=opt.batch_size, shuffle=True)

for epoch in range(opt.n_epochs): 
    prev_time = time.time() 
    for i, ((imgs1, _), (imgs2, _)) in enumerate(zip(dataloader1, dataloader2)):
        #imgs1 :MNIST(batch,1,28,28) / img2 :MINSTM(batch,3,32,32)
        
        batch_size = imgs1.shape[0]
        valid = torch.ones((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((batch_size,1),dtype = torch.float32 , device=device ,requires_grad=False)
        
        if imgs1.size(1) == 1:
            imgs1 = imgs1.expand(-1, 3, -1, -1) #(batch,3,28,28)
        imgs1 = F.interpolate(imgs1, size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)
        # (batch,1,28,28) into (batch,3,32,32)
        imgs1 = imgs1.to(dtype = torch.float32, device = device)
        imgs2 = imgs2.to(dtype = torch.float32, device = device)
        
        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        z = torch.tensor(np.random.normal(0,1,(batch_size,opt.latent_dim)) , dtype=torch.float32, device=device, requires_grad=False)
        gen_imgs1, gen_imgs2 = coupled_generators(z)

        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2
        g_loss.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminators
        # ----------------------
        optimizer_D.zero_grad()

        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

        d_loss = (
              adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity1_fake, fake)    
            + adversarial_loss(validity2_fake, fake)
        ) / 4

        d_loss.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {d_loss.item():.4f}][G loss: {g_loss.item():.4f}]--{time1round}")

    if epoch % 50 == 0 or epoch == 199:
        sample_images(row=10,imgtag=epoch)

