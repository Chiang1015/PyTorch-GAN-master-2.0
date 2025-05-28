#UNIT :Unsupervised Image-to-Image Translation Networks 
#無監督的圖像轉換
# use the trained model in 'models'
import torch
import glob
import time
import argparse
import itertools
import datetime
import random

import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=199, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="Apple2orange", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
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

#簡化版KL散度:懲罰潛在空間平均數遠離0
def compute_kl(mu):
    return torch.mean(torch.pow(mu, 2))

def sample_image(imgtag):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    X1 = imgs["A"].to(dtype=torch.float32,device=device)#apple
    X2 = imgs["B"].to(dtype=torch.float32,device=device)#orange
    _, Z1 = E1(X1)
    _, Z2 = E2(X2)
    fake_X1 = G1(Z2)
    fake_X2 = G2(Z1)
    img_sample = torch.cat((X1.data, fake_X2.data, X2.data, fake_X1.data), 0)
    # == real_apple
    # == pretend  
    # == real_orange
    # == pretend  
    save_image(img_sample, f'./samples/UNIT_{imgtag}.png', nrow=5, normalize=True)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

#input:(b,256,h,w) => (b,256,h,w)
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

#input:(b,3,256,256) => mu(b,256,64,64)/z(b,256,64,64)noise
class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, shared_block=None):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),#(h,w)+6
            nn.Conv2d(in_channels, dim, 7),#(h,w)-6
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        #(b,64,256,256)
        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2
        #(b,256,64,64)
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)
        self.shared_block = shared_block

    def reparameterization(self, mu):
        z = torch.randn(mu.shape, device=device) #noise
        return z + mu

    def forward(self, x):
        x = self.model_blocks(x) #(b,256,64,64)

        mu = self.shared_block(x)
        z = self.reparameterization(mu)
        return mu, z
    
#input: (b,256,64,64) => (b,3,256,256)
class Generator(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_upsample=2, shared_block=None):
        super(Generator, self).__init__()

        self.shared_block = shared_block

        layers = []
        dim = dim * 2 ** n_upsample#256
        # Residual blocks
        for _ in range(3):
            layers += [ResidualBlock(dim)]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_block(x) 
        x = self.model_blocks(x) # => (b,3,256,256)
        return x 
    
#input:(b,3,256,256)=>val(b,1,16,16)
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 3, padding=1)
        )
    def forward(self, img):
        return self.model(img)
    
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(root+f'/{mode}A' + "/*.*"))
        self.files_B = sorted(glob.glob( root+f'/{mode}B' + "/*.*"))
        
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
input_shape = (opt.channels, opt.img_height, opt.img_width)#(3,256,256)
shared_dim = opt.dim * 2 ** opt.n_downsample #(256)

# fix : share block may cause error
#object
shared_E1 = ResidualBlock(features=shared_dim).to(device)
shared_E2 = ResidualBlock(features=shared_dim).to(device)
E1 = Encoder(shared_block=shared_E1).to(device)
E2 = Encoder(shared_block=shared_E2).to(device)

shared_G1 = ResidualBlock(features=shared_dim).to(device)
shared_G2 = ResidualBlock(features=shared_dim).to(device)
G1 = Generator(shared_block=shared_G1).to(device)
G2 = Generator(shared_block=shared_G2).to(device)
D1 = Discriminator(input_shape).to(device)
D2 = Discriminator(input_shape).to(device)
#loss
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),lr=opt.lr,betas=(opt.b1, opt.b2))
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#unaligned : pick B_set  randomly
dataloader = DataLoader(
    ImageDataset(f"./Resorces/{opt.dataset_name}", transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
val_dataloader = DataLoader(
    ImageDataset(f"./Resorces/{opt.dataset_name}" , transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=0,
)
# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Initialize weights
if opt.epoch == 0:
    E1.apply(custom_weights_init)
    E2.apply(custom_weights_init)
    G1.apply(custom_weights_init)
    G2.apply(custom_weights_init)
    D1.apply(custom_weights_init)
    D2.apply(custom_weights_init)
else :
    E1.load_state_dict(torch.load(f"./models/UNIT_E1_{opt.epoch}"))
    E2.load_state_dict(torch.load(f"./models/UNIT_E2_{opt.epoch}"))
    G1.load_state_dict(torch.load(f"./models/UNIT_G1_{opt.epoch}"))
    G2.load_state_dict(torch.load(f"./models/UNIT_G2_{opt.epoch}"))
    D1.load_state_dict(torch.load(f"./models/UNIT_D1_{opt.epoch}"))
    D2.load_state_dict(torch.load(f"./models/UNIT_D2_{opt.epoch}"))

# Loss weights
lambda_0 = 10   # GAN
lambda_1 = 0.1  # KL (encoded images/encoded translated images)
lambda_2 = 100  # ID pixel-wise
lambda_3 = 100  # Cycle pixel-wise

for epoch in range(opt.epoch+1, opt.n_epochs):
    prev_time = time.time()
    for i, batch in enumerate(dataloader):
        X1 = batch["A"].to(dtype=torch.float32,device=device)#apple
        X2 = batch["B"].to(dtype=torch.float32,device=device)#orange
        #(b,1,16,16)
        valid = torch.ones((X1.size(0), *D1.output_shape),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((X1.size(0), *D1.output_shape),dtype = torch.float32 , device=device ,requires_grad=False)

        # ---------------
        #  Train E and G
        # ---------------
        optimizer_G.zero_grad()
        mu1, Z1 = E1(X1)
        mu2, Z2 = E2(X2)
        # Reconstruct images
        recon_X1 = G1(Z1)
        recon_X2 = G2(Z2)
        # Translate images
        fake_X1 = G1(Z2)
        fake_X2 = G2(Z1)
        # Cycle translation
        mu1_, Z1_ = E1(fake_X1)
        mu2_, Z2_ = E2(fake_X2)
        cycle_X1 = G1(Z2_)
        cycle_X2 = G2(Z1_)
        # Loss_GAN:真實度
        loss_GAN = criterion_GAN(D1(fake_X1), valid) + criterion_GAN(D2(fake_X2), valid)
        # Loss_KL:強制 latent 分佈近似標準常態分佈
        loss_KL = compute_kl(mu1) + compute_kl(mu2) + compute_kl(mu1_) + compute_kl(mu2_)
        # Loss_ID:重塑能力
        loss_ID = criterion_pixel(recon_X1, X1) + criterion_pixel(recon_X2, X2)
        # Loss_cyc: 穩定度(循環一致性損失，確保跨域轉換後能還原回原圖)
        loss_cyc = criterion_pixel(cycle_X1, X1) + criterion_pixel(cycle_X2, X2)
        # lambda:10/0.1/100/100
        loss_G = (lambda_0 * loss_GAN) + (lambda_1 * loss_KL) + (lambda_2 * loss_ID) + (lambda_3*loss_cyc)
        loss_G.backward()
        optimizer_G.step()

        # D affect Loss_GAN
        # -----------------------
        #  Train Discriminator 1
        # -----------------------
        optimizer_D1.zero_grad()
        loss_D1 = criterion_GAN(D1(X1), valid) + criterion_GAN(D1(fake_X1.detach()), fake)
        loss_D1.backward()
        optimizer_D1.step()
         # -----------------------
        #  Train Discriminator 2
        # -----------------------
        optimizer_D2.zero_grad()
        loss_D2 = criterion_GAN(D2(X2), valid) + criterion_GAN(D2(fake_X2.detach()), fake)
        loss_D2.backward()
        optimizer_D2.step()

    time1round = datetime.timedelta(seconds=time.time()-prev_time)
    print(f'[Epoch:{epoch:03d}/{opt.n_epochs:03d}][D loss:{(loss_D1 + loss_D2).item():.4f}][G loss:{loss_G.item():.4f}]',end='')
    print(f'--{time1round}')

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if epoch % 50 == 0:
        sample_image(imgtag=epoch)
    if epoch % 199 == 0:
        torch.save(E1.state_dict(), f"./models/UNIT_E1_{epoch}")
        torch.save(E2.state_dict(), f"./models/UNIT_E2_{epoch}")
        torch.save(G1.state_dict(), f"./models/UNIT_G1_{epoch}")
        torch.save(G2.state_dict(), f"./models/UNIT_G2_{epoch}")
        torch.save(D1.state_dict(), f"./models/UNIT_D1_{epoch}")
        torch.save(D2.state_dict(), f"./models/UNIT_D2_{epoch}")
        
for i in range(5):
    sample_image(imgtag=f'final_{i}')
