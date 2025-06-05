#Enhanced Super-Resolution GAN
#ESRGAN改進了原始的SRGAN架構，採用了RRDB(Residual-in-Residual Dense Block)
# This model's training is heavy loading for household device, recommend to put it to Kaggle or Colab
import torch
import argparse
import torch.nn as nn
import numpy as np
import glob
import time
import datetime

from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset #自訂集
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import vgg19,VGG19_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=46, help="epoch to start training from") # If you want to retrain, default = 0

parser.add_argument("--n_epochs", type=int, default=47, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="celebA", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hr_shape = (opt.hr_height, opt.hr_width) #(256,256)

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

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    #從-1~1還原0~255
    for c in range(3): #c:dim
        tensors[:, c].mul_(std[c]).add_(mean[c]) #tensor*std + mean
    return torch.clamp(tensors, 0, 255)

def sample_images(imgtag,sample):
    generator.eval()
    with torch.inference_mode():
        img_lr = nn.functional.interpolate(sample, scale_factor=4)#方程式補值
        gen_hr = generator(sample)#GAN補值

    img_grid = denormalize(torch.cat((img_lr,gen_hr),-1))
    save_image(img_grid, f'./samples/ESRGAN_{imgtag}.png', nrow=1, normalize=False)
    generator.train()

# Vgg19前35層
# input:(batch,3,x,x) => (batch,512,x/32,x/32)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

# obj=DenseResidualBlock(channel)
# shape unchange
class DenseResidualBlock(nn.Module):
    # Core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)

        #channel ratio(inputs:out) : 1st 1:1 / 2nd 2:1 ...5th 5:1
        # out *0.2 + 原始
        return out.mul(self.res_scale) + x
# obj=ResidualInResidualDenseBlock(channel)
# shape unchange   
class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

#input:(batch(3,x,x) => img(batch,3,4x,4x)
#(channels, filters=64, num_res_blocks=16, num_upsample=2)
class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers(channel*0.25,size*2)
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2), #(batch, C, H, W)=>(batch, C // r², H × r, W × r)
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x) #(batch,64,256,256)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)#逐元素相加
        #(batch,64,256,256)
        out = self.upsampling(out) #(batch,4,256*4,256*4)
        out = self.conv3(out) #(batch,3,256*4,256*4)
        return out

#input :(batch,3,256,256) => no sig(batch,1,16,16)
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)#(1,16,16)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels #3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        #(batch,512,16,16)
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        #channel :3->64->64->128->128->256->256->512->512->1
        #size    :256->256->128->128->64->64->32->32->16->16
        return self.model(img)


# Normalization parameters for pre-trained PyTorch models
#配合Vgg19預訓練的圖片處理
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# {"lr": img_lr, "hr": img_hr} 
class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose( #(64,64)
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(#(256,256)
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB") #如圖片是.png有加透明通道為4通道需.convert('RGB')，.jpg不用
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)



#object
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)

feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()
#Loss
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
#optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = DataLoader(
    ImageDataset("./Resorces/face64X64/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)

if opt.epoch != 0:
    generator.load_state_dict(torch.load(f"./models/ersG_{opt.epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"./models/ersD_{opt.epoch}.pth"))
else:
    generator.apply(custom_weights_init)
    discriminator.apply(custom_weights_init)

for epoch in range(opt.epoch+1, opt.n_epochs):
    prev_time = time.time()
    for i, imgs in enumerate(dataloader):
        imgs_lr = imgs["lr"].to(dtype=torch.float32,device=device)
        imgs_hr = imgs["hr"].to(dtype=torch.float32,device=device)
        #discriminator.output_shape = (1,16,16)
        valid = torch.ones((imgs_lr.size(0),*discriminator.output_shape),dtype = torch.float32 , device=device ,requires_grad=False)
        fake =  torch.zeros((imgs_lr.size(0),*discriminator.output_shape),dtype = torch.float32 , device=device ,requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        batches_done = epoch * len(dataloader) + i
        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][G pixel: {loss_pixel.item():.4f}]")
            continue

        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)
        #relativistic average GAN
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)#(batch,512,8,8)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss (_adv = 0.005 / _pixel = 0.01)
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        #這張真圖是否比假圖還真？這張假圖是否比真圖還假？
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
    time1round = datetime.timedelta(seconds = time.time() - prev_time)
    #_pixel:上採樣能力 / _gan: 仿真能力 / _content:仿特徵能力
    print(f"[Epoch {epoch:03d}/{opt.n_epochs:03d}][D loss: {loss_D.item():.4f}][G pixel: {loss_G.item():.4f}]",end='')
    print(f'[0.01_pixel: {loss_pixel.item():.4f}][0.005_gan:{loss_GAN.item():.4f}][1_content:{loss_content.item():.4f}]--{time1round}')

    if epoch % 2 == 0:
        torch.save(generator.state_dict(), f"./models/ersG_{epoch}.pth" )
        torch.save(discriminator.state_dict(), f"./models/ersD_{epoch}.pth" )

    if epoch % 50 == 0 or epoch == 199:
        sample_images(imgtag=epoch,sample = imgs_lr)


for i in range(5):
    imgs = next(iter(dataloader))
    imgs_lr = imgs['lr'].to(dtype=torch.float32,device=device)
    sample_images(imgtag=f'final_{i}',sample = imgs_lr)
